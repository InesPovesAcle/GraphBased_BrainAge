#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd

results_dir = Path("/mnt/newStor/paros/paros_WORK/ines/results/icvregression")
results_dir.mkdir(parents=True, exist_ok=True)

def read_table_auto(path: str) -> pd.DataFrame:
    """
    Read CSV/TSV/TXT with delimiter+encoding tolerance.
    Read Excel .xlsx/.xls with read_excel.
    """
    ext = Path(path).suffix.lower()

    # Excel
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    # Text
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep="\t", encoding="cp1252")


def regress_icv_correct_wide(
    regional_file: str,
    icv_file: str,
    out_file: str,
    thr_min_n: int = 10,
    drop_roi0: bool = True,
):
    # ---- load ----
    df = read_table_auto(regional_file)
    icv = read_table_auto(icv_file)

    # ---- clean column names ----
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    icv.columns = icv.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # ---- force ROI column name ----
    df = df.rename(columns={df.columns[0]: "ROI"})
    roi_col = "ROI"

    # ---- optional: drop ROI==0 (background/unknown) ----
    if drop_roi0:
        df[roi_col] = pd.to_numeric(df[roi_col], errors="coerce")
        df = df[df[roi_col] != 0].copy()

    # ---- identify ID column in ICV ----
    id_col = "file" if "file" in icv.columns else ("ID" if "ID" in icv.columns else icv.columns[0])

    # ---- ensure ICV is in mm3 ----
    icv = icv.copy()
    if "icv_mm3" in icv.columns:
        icv["icv_mm3"] = pd.to_numeric(icv["icv_mm3"], errors="coerce")
    elif "icv_ml" in icv.columns:
        icv["icv_ml"] = pd.to_numeric(icv["icv_ml"], errors="coerce")
        icv["icv_mm3"] = icv["icv_ml"] * 1000.0
    elif "n_voxels_gt0" in icv.columns:
        icv["n_voxels_gt0"] = pd.to_numeric(icv["n_voxels_gt0"], errors="coerce")
        icv["icv_mm3"] = icv["n_voxels_gt0"]  # vox=1mm3
    else:
        raise KeyError(
            "ICV file must contain one of: icv_mm3, icv_ml, n_voxels_gt0. "
            f"Columns: {list(icv.columns)}"
        )

    # ---- subject mapping (extract Sxxxx) ----
    icv["subject"] = icv[id_col].astype(str).str.extract(r"(S\d+)")
    icv = icv.dropna(subset=["subject", "icv_mm3"])
    icv = icv[icv["icv_mm3"] > 0].copy()
    icv = icv.drop_duplicates(subset=["subject"], keep="first")
    icv_map = dict(zip(icv["subject"], icv["icv_mm3"]))

    # ---- sample columns in regional (contain Sxxxx) ----
    sample_cols = [c for c in df.columns[1:] if re.search(r"(S\d+)", str(c))]
    if not sample_cols:
        raise ValueError(
            "No sample columns detected. This usually means the delimiter was wrong.\n"
            f"Detected columns: {list(df.columns)[:10]} ..."
        )

    # ---- NEW: keep only ONE column per subject (first seen) ----
    seen = set()
    sample_cols_unique = []
    for c in sample_cols:
        m = re.search(r"(S\d+)", str(c))
        if not m:
            continue
        s = m.group(1)
        if s in seen:
            continue
        seen.add(s)
        sample_cols_unique.append(c)
    sample_cols = sample_cols_unique

    # ---- long format ----
    long = df.melt(id_vars=[roi_col], value_vars=sample_cols, var_name="sample", value_name="roi_vol")
    long["subject"] = long["sample"].astype(str).str.extract(r"(S\d+)")
    long["icv"] = long["subject"].map(icv_map)

    # ---- numeric + drop missing ----
    long["roi_vol"] = pd.to_numeric(long["roi_vol"], errors="coerce")
    long["icv"] = pd.to_numeric(long["icv"], errors="coerce")
    long = long.dropna(subset=["roi_vol", "icv"]).copy()

    mean_icv = float(long["icv"].mean())

    # ---- compute beta per ROI ----
    g = long.groupby(roi_col, sort=False)
    n = g["icv"].size()
    var_x = g["icv"].var(ddof=1)

    mx = g["icv"].mean()
    my = g["roi_vol"].mean()
    cov_xy = g.apply(
        lambda d: ((d["icv"] - mx.loc[d.name]) * (d["roi_vol"] - my.loc[d.name])).sum() / (len(d) - 1)
    )

    beta = cov_xy / var_x
    beta = beta.where((n >= thr_min_n) & (var_x != 0), np.nan)

    long = long.merge(beta.rename("beta_icv"), left_on=roi_col, right_index=True, how="left")

    # ---- regression correction ----
    long["roi_icv_corrected_mm3"] = long["roi_vol"] - long["beta_icv"] * (long["icv"] - mean_icv)

    # ---- back to wide ----
    wide_corr = (
        long.pivot_table(index=roi_col, columns="sample", values="roi_icv_corrected_mm3", aggfunc="mean")
        .reset_index()
    )

    # ---- save ----
    wide_corr.to_csv(out_file, index=False)

    # ---- QC prints ----
    print("Saved:", out_file)
    print("ROI column:", roi_col)
    print("Regional shape (after drop_roi0 if applied):", df.shape)
    print("Sample columns used (unique subjects):", len(sample_cols))
    print("Unique subjects matched:", long["subject"].nunique())
    print("Mean ICV used (mm3):", mean_icv)
    print("Output shape:", wide_corr.shape)
    print("Non-NaN per ROI (desc):")
    print(wide_corr.drop(columns=[roi_col]).notna().sum(axis=1).describe())


if __name__ == "__main__":
    regional_file = "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.txt"
    icv_file      = "/mnt/newStor/paros/paros_WORK/ines/results/icv_from_masks.xlsx"
    out_file      = "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_ICVregressed_mm3_uniqueSubj.csv"

    regress_icv_correct_wide(regional_file, icv_file, out_file)
    