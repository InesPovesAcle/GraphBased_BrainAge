#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:31:56 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


# =========================================================
# BASE PATH
# =========================================================
BASE = Path("/mnt/newStor/paros/paros_WORK/ines/data")


# =========================================================
# DATASET CONFIG
# =========================================================
DATASETS = {
    "ADNI": {
        "regional_file": BASE / "Regional_stats/ADNI/ADNI_studywide_stats_for_volume.txt",
        "brain_mask_dir": BASE / "brain_masks/ADNI",
        "subject_prefix": "R",
        "out_dir": BASE / "Regional_stats/ADNI",
    },
    "ADDecode": {
        "regional_file": BASE / "Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.txt",
        "brain_mask_dir": BASE / "brain_masks/ADDecode",
        "subject_prefix": "S",
        "out_dir": BASE / "Regional_stats/ADDecode",
    },
    "ADRC": {
        "regional_file": BASE / "Regional_stats/ADRC/ADRC_studywide_stats_for_volume.txt",
        "brain_mask_dir": BASE / "brain_masks/ADRC",
        "subject_prefix": "D",
        "out_dir": BASE / "Regional_stats/ADRC",
    },
    "HABS": {
        "regional_file": BASE / "Regional_stats/HABS/HABS_studywide_stats_for_volume.txt",
        "brain_mask_dir": BASE / "brain_masks/HABS",
        "subject_prefix": "H",
        "out_dir": BASE / "Regional_stats/HABS",
    },
}

DATASETS = {"ADDecode": {
    "regional_file": BASE / "Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.txt",
    "brain_mask_dir": BASE / "brain_masks/ADDecode",
    "subject_prefix": "S",
    "out_dir": BASE / "Regional_stats/ADDecode",
}
    }

# outliers 01402, 04129, 04086, 04300, 01277, 01257, 04472, 01516, 01501, 01541, 04602

# True => percentages, False => proportions
MULTIPLY_BY_100 = True


# =========================================================
# HELPERS
# =========================================================
def read_table_auto(path: str | Path) -> pd.DataFrame:
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except Exception:
            pass

    raise ValueError(f"Could not read file: {path}")


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def extract_subject_id(text, subject_prefix: str) -> str | None:
    """
    Extract IDs like:
      R0072_y0   for ADNI
      S1234      for ADDecode
      D1234      for ADRC
      H1234      for HABS

    Also supports visit suffixes when present:
      S1234_y0, D1234_y2, H1234_y1, etc.
    """
    if pd.isna(text):
        return None

    text = str(text).strip()
    pattern = rf"({re.escape(subject_prefix)}\d+(?:_y\d+)?)"
    m = re.search(pattern, text)
    return m.group(1) if m else None


def find_structure_col(df: pd.DataFrame) -> str:
    for c in ["structure", "Structure", "label", "Label", "name", "Name"]:
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find structure-name column. "
        f"Columns found: {list(df.columns[:10])}"
    )


def find_row_idx(df: pd.DataFrame, structure_col: str, structure_name: str):
    rows = df.index[
        df[structure_col].astype(str).str.strip().str.lower() == structure_name.strip().lower()
    ].tolist()
    return rows[0] if rows else None


def detect_sample_cols(df: pd.DataFrame, subject_prefix: str) -> list[str]:
    """
    Tries all columns except the first two metadata columns, then keeps those
    whose names contain valid subject IDs.
    """
    return [
        c for c in df.columns[2:]
        if extract_subject_id(c, subject_prefix) is not None
    ]


def build_reference_map_from_masks(brain_mask_dir: str | Path, subject_prefix: str):
    """
    Returns
    -------
    brain_vol_map : dict
        subject -> total brain volume in mm^3
    voxel_info_map : dict
        subject -> metadata including voxel geometry and brain volume
    """
    brain_mask_dir = Path(brain_mask_dir)
    if not brain_mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {brain_mask_dir}")

    mask_files = sorted(
        list(brain_mask_dir.glob("*.nii")) +
        list(brain_mask_dir.glob("*.nii.gz")) +
        list(brain_mask_dir.glob("*.mgz"))
    )

    if not mask_files:
        raise ValueError(f"No mask files found in {brain_mask_dir}")

    brain_vol_map = {}
    voxel_info_map = {}
    duplicates = []

    print(f"Found {len(mask_files)} mask files in {brain_mask_dir}\n")

    for i, f in enumerate(mask_files, 1):
        subj = extract_subject_id(f.name, subject_prefix)
        if subj is None:
            continue

        img = nib.load(str(f))
        data = np.asanyarray(img.dataobj)

        zooms = img.header.get_zooms()
        dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
        voxel_volume_mm3 = dx * dy * dz

        n_voxels_gt0 = int(np.count_nonzero(data))
        brain_volume_mm3 = n_voxels_gt0 * voxel_volume_mm3
        brain_volume_cm3 = brain_volume_mm3 / 1000.0

        if n_voxels_gt0 <= 0:
            continue

        if subj in brain_vol_map:
            duplicates.append(subj)
            continue

        brain_vol_map[subj] = brain_volume_mm3
        voxel_info_map[subj] = {
            "dx_mm": dx,
            "dy_mm": dy,
            "dz_mm": dz,
            "voxel_volume_mm3": voxel_volume_mm3,
            "n_voxels_gt0": n_voxels_gt0,
            "brain_volume_mm3": brain_volume_mm3,
            "brain_volume_cm3": brain_volume_cm3,
            "mask_file": f.name,
            "mask_shape": tuple(data.shape),
        }

        print(
            f"[{i}/{len(mask_files)}] {subj} | "
            f"shape={tuple(data.shape)} | "
            f"dim=({dx:.4f}, {dy:.4f}, {dz:.4f}) mm | "
            f"voxel_volume={voxel_volume_mm3:.4f} mm^3 | "
            f"n_voxels_gt0={n_voxels_gt0} | "
            f"brain_volume={brain_volume_mm3:.2f} mm^3 | "
            f"{brain_volume_cm3:.2f} cm^3"
        )

    if not brain_vol_map:
        raise ValueError(
            f"No valid subject IDs found in mask filenames for prefix '{subject_prefix}'."
        )

    if duplicates:
        print("\nWarning: duplicate mask IDs found, keeping first occurrence:")
        print(sorted(set(duplicates))[:20])

    return brain_vol_map, voxel_info_map


def run_dataset(dataset_name: str, cfg: dict, multiply_by_100: bool = True):
    print("\n" + "#" * 90)
    print(f"RUNNING DATASET: {dataset_name}")
    print("#" * 90)

    regional_file = cfg["regional_file"]
    brain_mask_dir = cfg["brain_mask_dir"]
    subject_prefix = cfg["subject_prefix"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_abs_file = out_dir / f"{dataset_name}_studywide_stats_BrainAbs.csv"
    out_pct_file = out_dir / f"{dataset_name}_studywide_stats_BrainPct.csv"
    qc_file = out_dir / f"{dataset_name}_studywide_stats_maskQC.csv"

    # -------- load regional table --------
    df = read_table_auto(regional_file)
    df = clean_cols(df)

    print("Regional file:", regional_file)
    print("First 10 columns:")
    print(df.columns[:10].tolist())

    structure_col = find_structure_col(df)
    sample_cols = detect_sample_cols(df, subject_prefix)

    print("\nDetected first 10 subject columns:")
    print(sample_cols[:10])

    if not sample_cols:
        raise ValueError(
            f"No subject columns detected for dataset {dataset_name}. "
            f"Expected IDs starting with '{subject_prefix}'."
        )

    # -------- mask-based brain volumes --------
    brain_vol_map, voxel_info_map = build_reference_map_from_masks(
        brain_mask_dir=brain_mask_dir,
        subject_prefix=subject_prefix
    )

    # -------- pre-QC --------
    print("\n" + "=" * 80)
    print(f"PRE-NORMALIZATION QC: {dataset_name}")
    print("=" * 80)

    top5_subject_cols = [c for c in sample_cols if extract_subject_id(c, subject_prefix) in brain_vol_map][:5]
    top5_subjects = [extract_subject_id(c, subject_prefix) for c in top5_subject_cols]

    print("\nTop 5 matched subjects:")
    print(top5_subjects)

    ext_idx = find_row_idx(df, structure_col, "Exterior")
    hip_idx = find_row_idx(df, structure_col, "Left_Hippocampus")
    amy_idx = find_row_idx(df, structure_col, "Left_Amygdala")

    print("\n[QC 1] Total brain volume from masks for top 5 subjects")
    for subj in top5_subjects:
        info = voxel_info_map[subj]
        print(
            f"{subj} | mask_file={info['mask_file']} | "
            f"shape={info['mask_shape']} | "
            f"dim=({info['dx_mm']:.4f}, {info['dy_mm']:.4f}, {info['dz_mm']:.4f}) mm | "
            f"voxel_volume={info['voxel_volume_mm3']:.4f} mm^3 | "
            f"n_voxels_gt0={info['n_voxels_gt0']} | "
            f"brain_volume_mm3={info['brain_volume_mm3']:.2f} | "
            f"brain_volume_cm3={info['brain_volume_cm3']:.2f}"
        )

    brain_vols = pd.Series(brain_vol_map, dtype=float)

    print("\n[QC 2] Overall brain-volume summary from masks")
    print(brain_vols.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("\n[QC 3] Exterior vs mask-based brain volume for top 5 subjects")
    if ext_idx is None:
        print("No 'Exterior' row found.")
    else:
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            ext_val = pd.to_numeric(df.loc[ext_idx, col], errors="coerce")
            mask_vol = brain_vol_map[subj]
            ratio = ext_val / mask_vol if pd.notna(ext_val) and mask_vol > 0 else np.nan
            print(
                f"{subj} | Exterior={ext_val:.2f} | "
                f"MaskBrain_mm3={mask_vol:.2f} | "
                f"MaskBrain_cm3={mask_vol / 1000.0:.2f} | "
                f"Exterior/Mask={ratio:.6f}"
            )

    print("\n[QC 4] Left_Hippocampus and Left_Amygdala raw + normalized for top 5 subjects")
    if hip_idx is None or amy_idx is None:
        print("Could not find Left_Hippocampus and/or Left_Amygdala rows.")
    else:
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            mask_vol = brain_vol_map[subj]

            hip_raw = pd.to_numeric(df.loc[hip_idx, col], errors="coerce")
            amy_raw = pd.to_numeric(df.loc[amy_idx, col], errors="coerce")

            if multiply_by_100:
                hip_norm = hip_raw / mask_vol * 100 if pd.notna(hip_raw) and mask_vol > 0 else np.nan
                amy_norm = amy_raw / mask_vol * 100 if pd.notna(amy_raw) and mask_vol > 0 else np.nan
                unit_str = "%"
            else:
                hip_norm = hip_raw / mask_vol if pd.notna(hip_raw) and mask_vol > 0 else np.nan
                amy_norm = amy_raw / mask_vol if pd.notna(amy_raw) and mask_vol > 0 else np.nan
                unit_str = "proportion"

            print(
                f"{subj} | "
                f"Hip_raw={hip_raw:.2f} -> Hip_norm={hip_norm:.6f} {unit_str} | "
                f"Amy_raw={amy_raw:.2f} -> Amy_norm={amy_norm:.6f} {unit_str} | "
                f"Brain={mask_vol:.2f} mm^3 ({mask_vol / 1000.0:.2f} cm^3)"
            )

    print("\n[QC 5] Sum of all ROI values except Exterior vs mask-based brain volume for top 5 subjects")
    if ext_idx is None:
        print("No 'Exterior' row found.")
    else:
        roi_mask = df[structure_col].astype(str).str.strip().str.lower() != "exterior"
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            roi_vals = pd.to_numeric(df.loc[roi_mask, col], errors="coerce")
            total_regions = roi_vals.sum(skipna=True)
            mask_vol = brain_vol_map[subj]
            ratio = total_regions / mask_vol if mask_vol > 0 else np.nan

            print(
                f"{subj} | Sum_ROIs_noExterior={total_regions:.2f} | "
                f"MaskBrain_mm3={mask_vol:.2f} | "
                f"MaskBrain_cm3={mask_vol / 1000.0:.2f} | "
                f"SumROIs/Mask={ratio:.6f}"
            )

    print("=" * 80 + "\n")

    # -------- build absolute table --------
    df_abs = df.copy()

    df_abs.iloc[0, 0] = -1
    df_abs.iloc[0, 1] = "Brain"

    missing_subjects = set()
    invalid_subjects = set()

    for col in sample_cols:
        subj = extract_subject_id(col, subject_prefix)
        brain_vol_mm3 = brain_vol_map.get(subj, np.nan)

        if pd.isna(brain_vol_mm3) or brain_vol_mm3 <= 0:
            invalid_subjects.add(subj)
            continue

        df_abs.loc[df_abs.index[0], col] = brain_vol_mm3

    for col in sample_cols:
        subj = extract_subject_id(col, subject_prefix)
        if subj not in brain_vol_map:
            missing_subjects.add(subj)

    # -------- build percentage/proportion table --------
    df_pct = df_abs.copy()
    factor = 100.0 if multiply_by_100 else 1.0

    for col in sample_cols:
        subj = extract_subject_id(col, subject_prefix)
        brain_vol_mm3 = brain_vol_map.get(subj, np.nan)

        if pd.isna(brain_vol_mm3) or brain_vol_mm3 <= 0:
            continue

        df_pct[col] = pd.to_numeric(df_abs[col], errors="coerce") / brain_vol_mm3 * factor

    brain_idx_pct = find_row_idx(df_pct, structure_col, "Brain")
    if brain_idx_pct is not None:
        for col in sample_cols:
            subj = extract_subject_id(col, subject_prefix)
            if subj in brain_vol_map:
                df_pct.loc[brain_idx_pct, col] = 100.0 if multiply_by_100 else 1.0

    # -------- save QC table --------
    qc_rows = []
    for subj, info in voxel_info_map.items():
        qc_rows.append({
            "subject": subj,
            "mask_file": info["mask_file"],
            "mask_shape": str(info["mask_shape"]),
            "dx_mm": info["dx_mm"],
            "dy_mm": info["dy_mm"],
            "dz_mm": info["dz_mm"],
            "voxel_volume_mm3": info["voxel_volume_mm3"],
            "n_voxels_gt0": info["n_voxels_gt0"],
            "brain_volume_mm3": info["brain_volume_mm3"],
            "brain_volume_cm3": info["brain_volume_cm3"],
        })

    qc_df = pd.DataFrame(qc_rows).sort_values("subject")

    # -------- post-QC --------
    print("\n" + "=" * 80)
    print(f"POST-NORMALIZATION QC: {dataset_name}")
    print("=" * 80)

    brain_idx_abs = find_row_idx(df_abs, structure_col, "Brain")
    hip_idx_pct = find_row_idx(df_pct, structure_col, "Left_Hippocampus")
    amy_idx_pct = find_row_idx(df_pct, structure_col, "Left_Amygdala")
    hip_idx_raw = find_row_idx(df, structure_col, "Left_Hippocampus")

    print("\n[Post QC A] Brain row values in absolute table for top 5 subjects")
    if brain_idx_abs is None:
        print("No 'Brain' row found in absolute table.")
    else:
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            brain_row_val = pd.to_numeric(df_abs.loc[brain_idx_abs, col], errors="coerce")
            mask_vol_mm3 = brain_vol_map[subj]
            print(
                f"{subj} | Brain_row={brain_row_val:.2f} mm^3 | "
                f"MaskBrain={mask_vol_mm3:.2f} mm^3 | "
                f"{mask_vol_mm3 / 1000.0:.2f} cm^3 | "
                f"Match={(abs(brain_row_val - mask_vol_mm3) < 1e-6)}"
            )

    print("\n[Post QC B] Normalized Left_Hippocampus and Left_Amygdala for top 5 subjects")
    if hip_idx_pct is None or amy_idx_pct is None:
        print("Could not find Left_Hippocampus and/or Left_Amygdala in normalized table.")
    else:
        unit_str = "%" if multiply_by_100 else "proportion"
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            hip_norm = pd.to_numeric(df_pct.loc[hip_idx_pct, col], errors="coerce")
            amy_norm = pd.to_numeric(df_pct.loc[amy_idx_pct, col], errors="coerce")
            print(
                f"{subj} | Hip_norm={hip_norm:.6f} {unit_str} | "
                f"Amy_norm={amy_norm:.6f} {unit_str}"
            )

    print("\n[Post QC C] Sum of normalized ROIs excluding Brain for top 5 subjects")
    roi_mask_pct = ~df_pct[structure_col].astype(str).str.strip().isin(["Brain", "Exterior"])
    unit_str = "%" if multiply_by_100 else "proportion"
    for col in top5_subject_cols:
        subj = extract_subject_id(col, subject_prefix)
        vals = pd.to_numeric(df_pct.loc[roi_mask_pct, col], errors="coerce")
        total_norm = vals.sum(skipna=True)
        print(f"{subj} | Sum_normalized_ROIs_excl_Brain={total_norm:.6f} {unit_str}")

    print("\n[Post QC D] Raw vs normalized hippocampus for top 5 subjects")
    if hip_idx_raw is None or hip_idx_pct is None:
        print("Could not compare raw vs normalized hippocampus.")
    else:
        for col in top5_subject_cols:
            subj = extract_subject_id(col, subject_prefix)
            raw_val = pd.to_numeric(df.loc[hip_idx_raw, col], errors="coerce")
            pct_val = pd.to_numeric(df_pct.loc[hip_idx_pct, col], errors="coerce")
            brain_vol_mm3 = brain_vol_map[subj]
            recomputed = raw_val / brain_vol_mm3 * factor
            print(
                f"{subj} | raw={raw_val:.2f} | "
                f"Brain={brain_vol_mm3:.2f} mm^3 ({brain_vol_mm3 / 1000.0:.2f} cm^3) | "
                f"stored={pct_val:.6f} | recomputed={recomputed:.6f}"
            )

    # -------- save outputs --------
    df_abs.to_csv(out_abs_file, index=False)
    df_pct.to_csv(out_pct_file, index=False)
    qc_df.to_csv(qc_file, index=False)

    print("\nSaved absolute-volume table to:")
    print(out_abs_file)

    print("\nSaved percentage/proportion table to:")
    print(out_pct_file)

    print("\nSaved mask QC table to:")
    print(qc_file)

    print("\nQC summary:")
    print("Input shape:", df.shape)
    print("Number of subject columns:", len(sample_cols))
    print("Unique subjects in table:", len({extract_subject_id(c, subject_prefix) for c in sample_cols}))
    print("Unique subjects with mask-based brain volume:", len(brain_vol_map))
    print("Scale:", "percent" if multiply_by_100 else "proportion")

    if missing_subjects:
        print("\nSubjects present in table but missing mask file (first 20):")
        print(sorted(missing_subjects)[:20])

    if invalid_subjects:
        print("\nSubjects with invalid brain volume (first 20):")
        print(sorted(invalid_subjects)[:20])

    print("\nAbsolute table preview:")
    print(df_abs.iloc[:10, :8])

    print("\nNormalized table preview:")
    print(df_pct.iloc[:10, :8])

    print("\nMask QC preview:")
    print(qc_df.head(10))


def main():
    for dataset_name, cfg in DATASETS.items():
        try:
            run_dataset(
                dataset_name=dataset_name,
                cfg=cfg,
                multiply_by_100=MULTIPLY_BY_100
            )
        except Exception as e:
            print("\n" + "!" * 90)
            print(f"FAILED DATASET: {dataset_name}")
            print(f"Reason: {e}")
            print("!" * 90 + "\n")


if __name__ == "__main__":
    main()