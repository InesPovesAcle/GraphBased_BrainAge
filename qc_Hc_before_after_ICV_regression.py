#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:59:59 2026

@author: ines
"""

# ---- Full QC snippet: ROI 17 (Left-Hippocampus) vs ICV, before & after correction ----
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ---------- paths ----------
raw_file  = "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.txt"
corr_file = "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_ICVregressed_mm3_uniqueSubj.csv"
icv_file  = "/mnt/newStor/paros/paros_WORK/ines/results/icv_from_masks.xlsx"

roi = 17  # Left-Hippocampus

# ---------- load ----------
df_raw  = pd.read_csv(raw_file, sep=None, engine="python")
df_corr = pd.read_csv(corr_file)
icv     = pd.read_excel(icv_file)

# ---------- clean headers ----------
df_raw.columns  = df_raw.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
df_corr.columns = df_corr.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
icv.columns     = icv.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

# ---------- make sure ROI column is named ROI ----------
df_raw  = df_raw.rename(columns={df_raw.columns[0]: "ROI"})
df_corr = df_corr.rename(columns={df_corr.columns[0]: "ROI"})

# ---------- build ICV map in mm3 ----------
# prefer icv_mm3; else convert icv_ml; else use n_voxels_gt0 (vox=1mm3)
id_col = "file" if "file" in icv.columns else ("ID" if "ID" in icv.columns else icv.columns[0])

icv = icv.copy()
icv["subject"] = icv[id_col].astype(str).str.extract(r"(S\d+)")
if "icv_mm3" in icv.columns:
    icv["icv_mm3"] = pd.to_numeric(icv["icv_mm3"], errors="coerce")
elif "icv_ml" in icv.columns:
    icv["icv_ml"] = pd.to_numeric(icv["icv_ml"], errors="coerce")
    icv["icv_mm3"] = icv["icv_ml"] * 1000.0
elif "n_voxels_gt0" in icv.columns:
    icv["n_voxels_gt0"] = pd.to_numeric(icv["n_voxels_gt0"], errors="coerce")
    icv["icv_mm3"] = icv["n_voxels_gt0"]
else:
    raise KeyError(f"ICV file missing icv_mm3/icv_ml/n_voxels_gt0. Columns: {list(icv.columns)}")

icv = icv.dropna(subset=["subject", "icv_mm3"])
icv = icv[icv["icv_mm3"] > 0].drop_duplicates(subset=["subject"], keep="first")
icv_map = dict(zip(icv["subject"], icv["icv_mm3"]))

# ---------- choose the EXACT subject columns used in df_corr (unique per subject) ----------
corr_subject_cols = [c for c in df_corr.columns if c != "ROI"]

# ---------- extract ROI 17 rows ----------
raw_row  = df_raw[df_raw["ROI"] == roi]
corr_row = df_corr[df_corr["ROI"] == roi]

if raw_row.empty:
    raise ValueError(f"ROI {roi} not found in raw table.")
if corr_row.empty:
    raise ValueError(f"ROI {roi} not found in corrected table.")

# ---------- build aligned vectors (ICV, raw ROI, corrected ROI) ----------
x_icv, y_raw, y_corr = [], [], []

for col in corr_subject_cols:
    m = re.search(r"(S\d+)", str(col))
    if not m:
        continue
    subj = m.group(1)

    icv_mm3 = icv_map.get(subj, np.nan)
    if np.isnan(icv_mm3):
        continue

    # raw table may have multiple columns per subject; pick the matching column name if present
    # best effort: use the same column name as corrected if it exists in raw; otherwise skip
    if col not in df_raw.columns:
        continue

    v_raw  = pd.to_numeric(raw_row[col].iloc[0], errors="coerce")
    v_corr = pd.to_numeric(corr_row[col].iloc[0], errors="coerce")

    if pd.isna(v_raw) or pd.isna(v_corr):
        continue

    x_icv.append(float(icv_mm3))
    y_raw.append(float(v_raw))
    y_corr.append(float(v_corr))

x_icv = np.array(x_icv)
y_raw = np.array(y_raw)
y_corr = np.array(y_corr)

print(f"Aligned N for ROI {roi}:", len(x_icv))

# ---------- correlations ----------
r_raw,  p_raw  = pearsonr(x_icv, y_raw)
r_corr, p_corr = pearsonr(x_icv, y_corr)

print(f"ROI {roi} correlation with ICV (raw):      r={r_raw:.3f}, p={p_raw:.3g}")
print(f"ROI {roi} correlation with ICV (corrected): r={r_corr:.3f}, p={p_corr:.3g}")

from pathlib import Path

qc_dir = Path("/mnt/newStor/paros/paros_WORK/ines/results/qc_hc_before_after_ICV_regression")
qc_dir.mkdir(parents=True, exist_ok=True)

# ---------- plots ----------
# ---------- plots ----------
plt.figure(figsize=(6,4))
plt.scatter(x_icv, y_raw)
plt.xlabel("ICV (mm³)")
plt.ylabel("ROI volume (raw, mm³)")
plt.title(f"ROI {roi} vs ICV (raw)")
plt.tight_layout()
plt.savefig(qc_dir / "QC_ROI17_raw.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(x_icv, y_corr)
plt.xlabel("ICV (mm³)")
plt.ylabel("ROI volume (ICV-corrected, mm³)")
plt.title(f"ROI {roi} vs ICV (corrected)")
plt.tight_layout()
plt.savefig(qc_dir / "QC_ROI17_corrected.png", dpi=300, bbox_inches="tight")
plt.show()

