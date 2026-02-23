#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

vol_path = os.path.join(
    os.environ["WORK"],
    "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.txt"
)

vol_path_norm = os.path.join(
    os.environ["WORK"],
    "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume_norm.txt"
)

# ----------------------------
# 1) Read volume table (mm3)
# ----------------------------
df = pd.read_csv(vol_path, sep="\t")

# Ensure ROI column name is exactly "ROI"
if df.columns[0] != "ROI":
    df = df.rename(columns={df.columns[0]: "ROI"})

# Subject columns are everything except ROI
subject_cols = [c for c in df.columns if c != "ROI"]

# Force numeric (prevents int/float assignment errors)
df[subject_cols] = df[subject_cols].apply(pd.to_numeric, errors="coerce").astype(float)

# ROI as numeric
df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce").astype("Int64")

# ----------------------------
# 2) Get total brain (ROI 0) per subject, in mm3
# ----------------------------
row0 = df.loc[df["ROI"] == 0, subject_cols]
if row0.empty:
    raise ValueError("ROI 0 not found. Cannot normalize to total brain.")

total_mm3 = row0.iloc[0].copy()  # Series: index=subject_cols
# Avoid divide-by-zero
total_mm3 = total_mm3.replace({0.0: np.nan})

# ----------------------------
# 3) Build normalized output:
#    - ROI 0 stored as mL
#    - All other ROIs stored as % of total brain
# ----------------------------
out = df.copy()

# ROI 0 -> mL
out.loc[out["ROI"] == 0, subject_cols] = (total_mm3 / 1000.0).values

# All other ROIs -> %
mask_other = out["ROI"] != 0
out.loc[mask_other, subject_cols] = (out.loc[mask_other, subject_cols].divide(total_mm3, axis=1) * 100.0)

# Optional: round for readability (doesn't change meaning)
out.loc[out["ROI"] == 0, subject_cols] = out.loc[out["ROI"] == 0, subject_cols].round(3)   # mL
out.loc[mask_other, subject_cols] = out.loc[mask_other, subject_cols].round(6)             # %

# ----------------------------
# 4) Save
# ----------------------------
out.to_csv(vol_path_norm, sep="\t", index=False)
print("Saved normalized file:", vol_path_norm)

# ----------------------------
# 5) Quick sanity checks
# ----------------------------
# Hippocampus % should be between ~0 and a few percent typically
# (depends on your atlas/definition, but it should NOT be ~10000)
for roi in [17, 53]:
    if (out["ROI"] == roi).any():
        vals = out.loc[out["ROI"] == roi, subject_cols].iloc[0]
        print(f"ROI {roi} pct: min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}")
    else:
        print(f"ROI {roi} not found in file.")

# Total brain mL sanity check
tb = out.loc[out["ROI"] == 0, subject_cols].iloc[0]
print(f"Total brain (mL): min={np.nanmin(tb):.1f}, max={np.nanmax(tb):.1f}")