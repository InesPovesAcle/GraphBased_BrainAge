#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
WORK = os.environ["WORK"]

COHORT_NAME = "ADNI"   # <<< CAMBIA SOLO ESTO

RESULTS_ROOT = os.path.join(WORK, "ines/results")

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI",
    "ADRC": "BrainAgePredictionADRC",
    "HABS": "BrainAgePredictionHABS",
    "AD_DECODE": "BrainAgePredictionADDECODE",
}

if COHORT_NAME not in RESULTS_DIR_MAP:
    raise ValueError(f"Unsupported COHORT_NAME: {COHORT_NAME}")

RESULTS_DIR = os.path.join(RESULTS_ROOT, RESULTS_DIR_MAP[COHORT_NAME])

# metadata con cBAG generado por training
# RAW aligned metadata: base correcta para validación clínica
RAW_METADATA_PATH = {
    "ADNI": os.path.join(WORK, "ines/results/harmonized/ADNI/graphs/adni_metadata_all_aligned_raw.csv"),
    "ADRC": os.path.join(WORK, "ines/results/harmonized/ADRC/graphs/adrc_metadata_all_aligned_raw.csv"),
    "HABS": os.path.join(WORK, "ines/results/harmonized/HABS/graphs/habs_metadata_all_aligned_raw.csv"),
    "AD_DECODE": os.path.join(WORK, "ines/results/harmonized/AD_DECODE/graphs/ad_decode_metadata_all_aligned_raw.csv"),
}[COHORT_NAME].strip()

# Full-cohort predictions from training
FULL_COHORT_PRED_PATH = {
    "ADNI": os.path.join(RESULTS_DIR, "adni_full_cohort_predictions.csv"),
    "ADRC": os.path.join(RESULTS_DIR, "adrc_full_cohort_predictions.csv"),
    "HABS": os.path.join(RESULTS_DIR, "habs_full_cohort_predictions.csv"),
    "AD_DECODE": os.path.join(RESULTS_DIR, "addecode_full_cohort_predictions.csv"),
}[COHORT_NAME].strip()

# BrainPct: hipocampo relativo
BRAINPCT_PATH = {
    "ADNI": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADNI/ADNI_studywide_stats_BrainPct.csv",
    "ADRC": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADRC/ADRC_studywide_stats_BrainPct.csv",
    "HABS": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/HABS/HABS_studywide_stats_BrainPct.csv",
    "AD_DECODE": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/ADDecode_studywide_stats_BrainPct.csv",
}[COHORT_NAME].strip()

# BrainAbs: volumen absoluto total de cerebro
BRAINABS_PATH = {
    "ADNI": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADNI/ADNI_studywide_stats_BrainAbs.csv",
    "ADRC": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADRC/ADRC_studywide_stats_BrainAbs.csv",
    "HABS": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/HABS/HABS_studywide_stats_BrainAbs.csv",
    "AD_DECODE": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/ADDecode_studywide_stats_BrainAbs.csv",
}[COHORT_NAME].strip()

# FA: FA del hipocampo
FA_PATH = {
    "ADNI": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADNI/ADNI_studywide_stats_for_fa.txt",
    "ADRC": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADRC/ADRC_studywide_stats_for_fa.txt",
    "HABS": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/HABS/HABS_studywide_stats_for_fa.txt",
    "AD_DECODE": "/mnt/newStor/paros/paros_WORK/ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_fa.txt",
}[COHORT_NAME].strip()

if COHORT_NAME == "AD_DECODE":
    out_stem = "addecode"
else:
    out_stem = COHORT_NAME.lower()

OUT_CSV = os.path.join(
    RESULTS_DIR,
    f"{out_stem}_metadata_all_raw_with_predictions_plus_brainvol_hipp_fa.csv"
).strip()

OUT_XLSX = os.path.join(
    RESULTS_DIR,
    f"{out_stem}_metadata_all_raw_with_predictions_plus_brainvol_hipp_fa.xlsx"
).strip()
OVERWRITE_INPUT_METADATA = False


# =========================================================
# HELPERS
# =========================================================
def load_table(path):
    path = str(path).strip()
    lower = path.lower()

    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".txt") or lower.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {path}")


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_structure_name(x):
    s = str(x).strip().upper()
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def merge_predictions_into_raw_metadata(metadata_df, pred_df):
    pred_df = pred_df.copy()

    if "graph_id" not in pred_df.columns:
        raise ValueError("Prediction file must contain 'graph_id'.")

    pred_df["merge_id"] = pred_df["graph_id"].astype(str).map(normalize_subject_id)

    merge_key, overlap = choose_best_merge_key(metadata_df, pred_df)

    if merge_key is None or overlap == 0:
        raise ValueError("Could not merge predictions into raw metadata: overlap=0")

    out = metadata_df.copy()
    out["_merge_id_tmp"] = out[merge_key].astype(str).map(normalize_subject_id)

    pred_keep_cols = [
        "merge_id",
        "graph_id",
        "age_true",
        "pred_raw",
        "pred_bias_corrected",
        "BAG_raw",
        "cBAG",
    ]
    pred_keep_cols = [c for c in pred_keep_cols if c in pred_df.columns]

    pred_small = pred_df[pred_keep_cols].drop_duplicates(subset=["merge_id"]).copy()

    out = out.merge(
        pred_small,
        left_on="_merge_id_tmp",
        right_on="merge_id",
        how="left"
    )
    out = out.drop(columns=["_merge_id_tmp", "merge_id"], errors="ignore")

    print(
        f"Predictions merged with merge_key={merge_key}, "
        f"overlap={overlap}, "
        f"non-null cBAG={out['cBAG'].notna().sum() if 'cBAG' in out.columns else 'NA'}"
    )
    return out


def normalize_subject_id(x):
    """
    Normaliza IDs para merge robusto entre metadata y studywide stats.

    Casos cubiertos:
    - HABS: H4369_y0 -> H4369_Y0
    - AD_DECODE studywide: S02110 -> 2110
    - ADRC studywide: D0007 -> 0007
    - ADRC metadata: ADRC0007 / 7 / 007 / 0007 -> 0007
    """
    s = str(x).strip().upper()
    s = re.sub(r"\.0$", "", s)

    # quitar sufijos
    s = re.sub(r"_MASTER_T$", "", s)
    s = re.sub(r"_MASTER$", "", s)
    s = re.sub(r"_T$", "", s)

    # mantener _Y0, _Y2, etc.
    s = re.sub(r"_Y(\d+)$", lambda m: f"_Y{m.group(1)}", s)

    # -----------------------------------------------------
    # AD_DECODE: S02110 -> 2110
    # -----------------------------------------------------
    m = re.fullmatch(r"S0*([0-9]+)", s)
    if m:
        return str(int(m.group(1)))

    # -----------------------------------------------------
    # ADRC studywide: D0007 -> 0007
    # -----------------------------------------------------
    m = re.fullmatch(r"D([0-9]+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    # -----------------------------------------------------
    # ADRC metadata: ADRC0007 -> 0007
    # -----------------------------------------------------
    m = re.fullmatch(r"ADRC0*([0-9]+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    # -----------------------------------------------------
    # ADRC metadata a veces puede venir solo como número
    # 7 / 007 / 0007 -> 0007
    # -----------------------------------------------------
    if COHORT_NAME == "ADRC" and re.fullmatch(r"[0-9]+", s):
        return s[-4:].zfill(4)

    return s


def build_possible_metadata_keys(df):
    candidates = [
        "Subject_ID",
        "graph_id",
        "connectome_key",
        "match_id",
        "subject_id",
        "PTID",
        "ptid",
        "RID",
        "ID",
        "MRI_Exam",
    ]
    return [c for c in candidates if c in df.columns]


def choose_best_merge_key(metadata_df, stats_long_df):
    stats_ids = set(stats_long_df["merge_id"].astype(str).tolist())

    best_col = None
    best_overlap = -1

    for col in build_possible_metadata_keys(metadata_df):
        tmp_ids = set(metadata_df[col].astype(str).map(normalize_subject_id).tolist())
        overlap = len(tmp_ids.intersection(stats_ids))
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    return best_col, best_overlap


def find_structure_column(df):
    return first_existing_column(df, ["structure", "Structure", "STRUCTURE", "Structure "])


def get_subject_columns(df):
    ignore = {
        "ROI", "roi", "Index2", "index",
        "structure", "Structure", "STRUCTURE", "Structure "
    }
    return [c for c in df.columns if c not in ignore]


def extract_row_values_as_long(df, structure_candidates, value_name, verbose=True):
    """
    Busca una fila por estructura y la convierte a largo:
    subject_raw | value_name | merge_id
    """
    structure_col = find_structure_column(df)
    if structure_col is None:
        raise KeyError("No structure/Structure column found.")

    tmp = df.copy()
    tmp["_structure_norm"] = tmp[structure_col].map(normalize_structure_name)

    candidates_norm = [normalize_structure_name(x) for x in structure_candidates]
    hit = tmp[tmp["_structure_norm"].isin(candidates_norm)].copy()

    if len(hit) == 0:
        if verbose:
            print(f"\nNo encontré estas estructuras para {value_name}: {structure_candidates}")
            print("Primeras estructuras disponibles:")
            print(tmp[structure_col].astype(str).head(60).tolist())
        raise ValueError(f"No encontré ninguna de estas estructuras: {structure_candidates}")

    row = hit.iloc[0]
    subject_cols = get_subject_columns(df)

    long_df = pd.DataFrame({
        "subject_raw": subject_cols,
        value_name: [row[c] for c in subject_cols]
    })

    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df["merge_id"] = long_df["subject_raw"].map(normalize_subject_id)

    return long_df


def merge_one_metric(metadata_df, metric_df, metric_name, required=True):
    merge_key, overlap = choose_best_merge_key(metadata_df, metric_df)

    if merge_key is None or overlap == 0:
        msg = f"No pude mergear {metric_name}: overlap=0"
        if required:
            raise ValueError(msg)
        print(f"Skipping {metric_name}: {msg}")
        return metadata_df.copy()

    out = metadata_df.copy()
    out["_merge_id_tmp"] = out[merge_key].astype(str).map(normalize_subject_id)

    metric_small = metric_df[["merge_id", metric_name]].drop_duplicates(subset=["merge_id"]).copy()
    out = out.merge(metric_small, left_on="_merge_id_tmp", right_on="merge_id", how="left")
    out = out.drop(columns=["_merge_id_tmp", "merge_id"], errors="ignore")

    print(
        f"{metric_name}: merge_key={merge_key}, "
        f"overlap={overlap}, non-null={out[metric_name].notna().sum()}"
    )
    return out


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print(f"COHORT_NAME = {COHORT_NAME}")
    print("=" * 80)

    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(f"Missing RAW metadata: {RAW_METADATA_PATH}")
    if not os.path.exists(FULL_COHORT_PRED_PATH):
        raise FileNotFoundError(f"Missing full-cohort predictions: {FULL_COHORT_PRED_PATH}")
    if not os.path.exists(BRAINPCT_PATH):
        raise FileNotFoundError(f"Missing BrainPct file: {BRAINPCT_PATH}")
    if not os.path.exists(BRAINABS_PATH):
        raise FileNotFoundError(f"Missing BrainAbs file: {BRAINABS_PATH}")
    if not os.path.exists(FA_PATH):
        raise FileNotFoundError(f"Missing FA file: {FA_PATH}")

    metadata_raw_df = load_table(RAW_METADATA_PATH)
    pred_df = load_table(FULL_COHORT_PRED_PATH)
    brainpct_df = load_table(BRAINPCT_PATH)
    brainabs_df = load_table(BRAINABS_PATH)
    fa_df = load_table(FA_PATH)
    
    metadata_df = merge_predictions_into_raw_metadata(metadata_raw_df, pred_df)
    
    print(f"metadata_raw_df shape: {metadata_raw_df.shape}")
    print(f"pred_df shape: {pred_df.shape}")
    print(f"metadata_df after prediction merge shape: {metadata_df.shape}")
    print(f"brainpct_df shape: {brainpct_df.shape}")
    print(f"brainabs_df shape: {brainabs_df.shape}")
    print(f"fa_df shape: {fa_df.shape}")

    # -----------------------------------------------------
    # HIPPOCAMPUS PCT from BrainPct
    # -----------------------------------------------------
    left_hipp_pct_df = extract_row_values_as_long(
        brainpct_df,
        structure_candidates=["Left_Hippocampus", "Left-Hippocampus"],
        value_name="Left_Hippocampus_pct"
    )

    right_hipp_pct_df = extract_row_values_as_long(
        brainpct_df,
        structure_candidates=["Right_Hippocampus", "Right-Hippocampus"],
        value_name="Right_Hippocampus_pct"
    )

    # -----------------------------------------------------
    # TOTAL BRAIN VOLUME from BrainAbs
    # -----------------------------------------------------
    total_brain_df = extract_row_values_as_long(
        brainabs_df,
        structure_candidates=["Brain"],
        value_name="Total_Brain_volume"
    )

    # -----------------------------------------------------
    # HIPPOCAMPUS FA from FA file
    # -----------------------------------------------------
    left_hipp_fa_df = extract_row_values_as_long(
        fa_df,
        structure_candidates=["Left_Hippocampus", "Left-Hippocampus"],
        value_name="Left_Hippocampus_FA"
    )

    right_hipp_fa_df = extract_row_values_as_long(
        fa_df,
        structure_candidates=["Right_Hippocampus", "Right-Hippocampus"],
        value_name="Right_Hippocampus_FA"
    )

    # -----------------------------------------------------
    # MERGES
    # -----------------------------------------------------
    merged_df = metadata_df.copy()

    merged_df = merge_one_metric(merged_df, total_brain_df, "Total_Brain_volume", required=False)
    merged_df = merge_one_metric(merged_df, left_hipp_pct_df, "Left_Hippocampus_pct", required=True)
    merged_df = merge_one_metric(merged_df, right_hipp_pct_df, "Right_Hippocampus_pct", required=True)
    merged_df = merge_one_metric(merged_df, left_hipp_fa_df, "Left_Hippocampus_FA", required=True)
    merged_df = merge_one_metric(merged_df, right_hipp_fa_df, "Right_Hippocampus_FA", required=True)

    # -----------------------------------------------------
    # DERIVED COLUMNS
    # -----------------------------------------------------
    merged_df["Hippocampus_Total_pct"] = (
        pd.to_numeric(merged_df["Left_Hippocampus_pct"], errors="coerce") +
        pd.to_numeric(merged_df["Right_Hippocampus_pct"], errors="coerce")
    )

    merged_df["Hippocampus_FA_Total"] = (
        pd.to_numeric(merged_df["Left_Hippocampus_FA"], errors="coerce") +
        pd.to_numeric(merged_df["Right_Hippocampus_FA"], errors="coerce")
    )

    merged_df["Hippocampus_FA_Mean"] = (
        merged_df[["Left_Hippocampus_FA", "Right_Hippocampus_FA"]]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )

    # -----------------------------------------------------
    # SANITY CHECKS
    # -----------------------------------------------------
    print("\n=== SANITY CHECKS ===")
    for c in [
        "Total_Brain_volume",
        "Left_Hippocampus_pct",
        "Right_Hippocampus_pct",
        "Hippocampus_Total_pct",
        "Left_Hippocampus_FA",
        "Right_Hippocampus_FA",
        "Hippocampus_FA_Total",
        "Hippocampus_FA_Mean",
    ]:
        if c in merged_df.columns:
            print(
                f"{c}: non-null={merged_df[c].notna().sum()} | "
                f"mean={pd.to_numeric(merged_df[c], errors='coerce').mean():.6f}"
            )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    merged_df.to_csv(OUT_CSV, index=False)
    merged_df.to_excel(OUT_XLSX, index=False)

    print(f"\nSaved merged CSV:  {OUT_CSV}")
    print(f"Saved merged XLSX: {OUT_XLSX}")

    if OVERWRITE_INPUT_METADATA:
        if RAW_METADATA_PATH.lower().endswith(".csv"):
            merged_df.to_csv(RAW_METADATA_PATH, index=False)
        elif RAW_METADATA_PATH.lower().endswith((".xlsx", ".xls")):
            merged_df.to_excel(RAW_METADATA_PATH, index=False)
        print(f"Overwritten input metadata: {RAW_METADATA_PATH}")


if __name__ == "__main__":
    main()