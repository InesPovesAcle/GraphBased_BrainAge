#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# UNIFIED COUNTING SUBJECTS STANDARD SCRIPT
# Change only: COHORT_NAME = "HABS" / "ADNI" / "ADRC" / "AD_DECODE"
# ============================================================

import os
import re
import glob
import zipfile
import random
import numpy as np
import pandas as pd
import torch

# =========================
# USER INPUT
# =========================
COHORT_NAME = "ADNI"   # "HABS", "ADNI", "ADRC", "AD_DECODE"
SAVE_OUTPUTS = True

# =========================
# REPRODUCIBILITY
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# =========================
# CONFIG
# =========================
CONFIG = {
    "HABS": {
        "connectome_mode": "folder_habs",
        "connectome_dir": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/HABS/connectomes/DWI/plain"
        ),
        "metadata_path": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/HABS/metadata/HABS_metadata.xlsx"
        ),
        "metadata_sheet": None,
        "subject_col": "Subject",
        "runno_col": "runno",
        "age_col": "Age",
        "sex_col": "Sex",
        "dx_col": "CDX_Cog",
        "apoe_col": "APOE4_Genotype",
    },

    "ADNI": {
        "connectome_mode": "folder_adni",
        "connectome_dir": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/ADNI/connectomes/DWI/plain"
        ),
        "metadata_path": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/ADNI/metadata/ADNI_metadata.xlsx"
        ),
        "metadata_sheet": None,
        "rid_col": "RID",
        "age_col": "AGE",
        "sex_col": "PTGENDER",
        "dx_col": "Research Group",
        "apoe_col1": "APOE_A1",
        "apoe_col2": "APOE_A2",
        "visit_col": "VISCODE",
    },

    "ADRC": {
        "connectome_mode": "folder_adrc",
        "connectome_dir": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/ADRC/connectomes/DWI/plain"
        ),
        "metadata_path": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/ADRC/metadata/ADRC_metadata.xlsx"
        ),
        "metadata_sheet": None,
        "metadata_id_col": "PTID",
        "age_col": "SUBJECT_AGE_SCREEN",
        "sex_col": "SUBJECT_SEX",
        "apoe_col": "APOE",
        "dx_flag_norm": "NORMCOG",
        "dx_flag_mci": "IMPNOMCI",
        "dx_flag_dem": "DEMENTED",
    },

    "AD_DECODE": {
        "connectome_mode": "zip_addecode",
        "zip_path": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/AD_DECODE/connectomes/AD_DECODE_connectome_act.zip"
        ),
        "zip_subdir": "connectome_act/",
        "metadata_path": os.path.join(
            os.environ["WORK"],
            "ines/data/harmonization/AD_DECODE/metadata/AD_DECODE_metadata.xlsx"
        ),
        "metadata_sheet": None,
        "metadata_mri_exam_col": "MRI_Exam",
        "age_col": "age",
        "sex_col": "sex",
        "dx_col": "Risk",
        "apoe_col": "genotype",
    }
}

if COHORT_NAME not in CONFIG:
    raise ValueError(f"Unknown cohort: {COHORT_NAME}. Use one of: {list(CONFIG.keys())}")

cfg = CONFIG[COHORT_NAME]

# =========================
# HELPERS
# =========================
def normalize_apoe_single(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper().replace("/", "").replace("-", "").replace(" ", "")

    m = re.match(r"APOE(\d)(\d)$", s)
    if m:
        a1, a2 = sorted([m.group(1), m.group(2)])
        return f"APOE{a1}{a2}"

    alleles = re.findall(r"[234]", s)
    if len(alleles) >= 2:
        a1, a2 = sorted([alleles[0], alleles[1]])
        return f"APOE{a1}{a2}"

    return s


def normalize_apoe_from_two_cols(a1, a2):
    a1 = pd.to_numeric(a1, errors="coerce")
    a2 = pd.to_numeric(a2, errors="coerce")

    out = []
    for x, y in zip(a1, a2):
        if pd.isna(x) or pd.isna(y):
            out.append(np.nan)
        else:
            x = int(x)
            y = int(y)
            if x in [2, 3, 4] and y in [2, 3, 4]:
                lo, hi = sorted([x, y])
                out.append(f"APOE{lo}{hi}")
            else:
                out.append(np.nan)
    return pd.Series(out)


def normalize_sex_generic(series):
    def _map(x):
        s = str(x).strip().lower()
        if s in ["m", "male", "1"]:
            return "Male (M)"
        if s in ["f", "female", "2"]:
            return "Female (F)"
        if s.startswith("m"):
            return "Male (M)"
        if s.startswith("f"):
            return "Female (F)"
        return "Unknown"
    return series.apply(_map)


def counts_perc(series, denom=None):
    s = series.dropna()
    counts = s.value_counts()
    if denom is None:
        denom = len(series)
    perc = (counts / denom * 100).round(1)
    return counts, perc


def save_if_requested(df, filename):
    if SAVE_OUTPUTS:
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")


def print_header(title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}\n")


def print_subheader(title):
    print(f"\n{'-'*40}")
    print(title)
    print(f"{'-'*40}")


def extract_4digit_match_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    groups = re.findall(r"(\d+)", s)
    if len(groups) == 0:
        return np.nan
    digits = "".join(groups)
    return digits[-4:].zfill(4)


def extract_adrc_connectome_match_id(filename):
    s = str(filename).strip().upper()

    m = re.search(r"D(\d+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    m = re.search(r"ADRC(\d+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    return extract_4digit_match_id(s)


# =========================
# LOAD CONNECTOMES
# =========================
print_header(f"{COHORT_NAME} CONNECTOMES")

connectome_index = None

# -------------------------
# HABS
# -------------------------
if cfg["connectome_mode"] == "folder_habs":
    connectome_dir = cfg["connectome_dir"]
    pattern = os.path.join(connectome_dir, "*_conn_plain.csv")
    files = sorted(glob.glob(pattern))

    print("Folder:", connectome_dir)
    print("Exists:", os.path.exists(connectome_dir))
    print(f"Found conn_plain files: {len(files)}")

    rows = []
    bad = 0

    for fp in files:
        base = os.path.basename(fp).replace("_conn_plain.csv", "")
        try:
            mat = pd.read_csv(fp, header=None)
            if mat.shape[0] == 0 or mat.shape[1] == 0:
                bad += 1
                continue
        except Exception:
            bad += 1
            continue

        rows.append({
            "CONN_KEY": base,
            "filepath": fp
        })

    connectome_index = pd.DataFrame(rows).drop_duplicates(subset=["CONN_KEY"])
    print(f"Valid conn_plain entries: {connectome_index.shape[0]}")

# -------------------------
# ADNI
# -------------------------
elif cfg["connectome_mode"] == "folder_adni":
    connectome_dir = cfg["connectome_dir"]
    files = []

    for root, _, fns in os.walk(connectome_dir):
        for fn in fns:
            if fn.endswith("_conn_plain.csv"):
                files.append(os.path.join(root, fn))

    print("Folder:", connectome_dir)
    print("Exists:", os.path.exists(connectome_dir))
    print(f"Found conn_plain files: {len(files)}")

    rows = []
    bad = 0

    for fp in files:
        fn = os.path.basename(fp).replace("_conn_plain.csv", "")
        m = re.match(r"^R(\d+)_y(\d+)$", fn)
        if not m:
            bad += 1
            continue

        rid = int(m.group(1))
        y = int(m.group(2))

        rows.append({
            "CONN_KEY": fn,
            "RID": rid,
            "Y": y,
            "filepath": fp
        })

    connectome_index = pd.DataFrame(rows).drop_duplicates(subset=["CONN_KEY"])
    print(f"Valid conn_plain entries: {connectome_index.shape[0]}")
    if bad > 0:
        print(f"Skipped (unexpected filename pattern): {bad}")

# -------------------------
# ADRC
# -------------------------
elif cfg["connectome_mode"] == "folder_adrc":
    connectome_dir = cfg["connectome_dir"]
    pattern = os.path.join(connectome_dir, "*_conn_plain.csv")
    files = sorted(glob.glob(pattern))

    print("Folder:", connectome_dir)
    print("Exists:", os.path.exists(connectome_dir))
    print(f"Found conn_plain files: {len(files)}")

    rows = []
    dropped_files = []

    for fp in files:
        ok = True
        mat = None
        base = os.path.basename(fp)

        try:
            mat = np.loadtxt(fp, delimiter=",")
        except Exception:
            try:
                df_tmp = pd.read_csv(fp)
                mat = df_tmp.values
            except Exception:
                ok = False

        if ok and isinstance(mat, np.ndarray):
            if mat.ndim != 2:
                ok = False
            if ok and mat.shape[0] != mat.shape[1]:
                ok = False
            if ok and np.isnan(mat).any():
                ok = False
        else:
            ok = False

        if not ok:
            dropped_files.append(fp)
            continue

        match_id = extract_adrc_connectome_match_id(base)
        if pd.isna(match_id):
            dropped_files.append(fp)
            continue

        rows.append({
            "CONN_KEY": match_id,
            "match_id": match_id,
            "filepath": fp,
            "filename": base
        })

    connectome_index = pd.DataFrame(rows)

    if connectome_index.empty:
        connectome_index = pd.DataFrame(columns=["CONN_KEY", "match_id", "filepath", "filename"])
    else:
        connectome_index = connectome_index.drop_duplicates(subset=["match_id"])

    print(f"Total connectomes after filtering: {connectome_index.shape[0]}")
    if len(dropped_files) > 0:
        print(f"Dropped (invalid matrix or ID parse fail): {len(dropped_files)}")
        print("Showing up to 5:")
        for x in dropped_files[:5]:
            print(" ", os.path.basename(x))

    if not connectome_index.empty:
        print("Example ADRC connectome keys:", connectome_index.head()["match_id"].tolist())

# -------------------------
# AD_DECODE
# -------------------------
elif cfg["connectome_mode"] == "zip_addecode":
    zip_path = cfg["zip_path"]
    zip_subdir = cfg["zip_subdir"]

    print("ZIP:", zip_path)
    print("Exists:", os.path.exists(zip_path))

    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        files = [
            f for f in z.namelist()
            if f.startswith(zip_subdir) and f.endswith("_conn_plain.csv")
        ]
        print(f"Found conn_plain files in ZIP: {len(files)}")

        for file in files:
            subject_raw = file.split("/")[-1].replace("_conn_plain.csv", "")
            if "_whitematter" in subject_raw:
                continue

            match = re.search(r"S(\d+)", subject_raw)
            if match:
                subject_id = match.group(1).zfill(5)
                rows.append({
                    "CONN_KEY": subject_id,
                    "filepath_in_zip": file
                })

    connectome_index = pd.DataFrame(rows).drop_duplicates(subset=["CONN_KEY"])
    print(f"Valid conn_plain entries after filtering: {connectome_index.shape[0]}")

else:
    raise ValueError(f"Unsupported connectome_mode: {cfg['connectome_mode']}")

# =========================
# CONNECTOME COUNTS
# =========================
print_subheader("CONNECTOME COUNTS")

if COHORT_NAME == "HABS":
    def parse_habs_runno(runno):
        m_sub = re.search(r"^H(\d+)", str(runno).strip(), flags=re.IGNORECASE)
        subj = m_sub.group(1).zfill(5) if m_sub else None

        m_tp = re.search(r"_y(\d+)", str(runno).strip(), flags=re.IGNORECASE)
        tp = f"y{m_tp.group(1)}" if m_tp else None
        return subj, tp

    parsed = [parse_habs_runno(x) for x in connectome_index["CONN_KEY"]]
    connectome_index["SUBJECT_ID"] = [p[0] for p in parsed]
    connectome_index["TIMEPOINT"] = [p[1] for p in parsed]

    unique_subjects = connectome_index["SUBJECT_ID"].dropna().nunique()
    print(f"Unique subjects: {unique_subjects}")

    tp_counts = connectome_index["TIMEPOINT"].value_counts()
    print("\nConnectomes per timepoint:")
    for tp, n in tp_counts.items():
        print(f"{tp}: {n}")

    tp_per_subj = connectome_index.groupby("SUBJECT_ID")["TIMEPOINT"].nunique()
    print("\n#timepoints per subject:")
    for k, v in tp_per_subj.value_counts().sort_index().items():
        print(f"{k} timepoints: {v} subjects")

elif COHORT_NAME == "ADNI":
    unique_subjects = connectome_index["RID"].nunique()
    print(f"Unique subjects (RID): {unique_subjects}")

    tp_counts = connectome_index["Y"].value_counts().sort_index()
    print("\nConnectomes per timepoint:")
    for y, n in tp_counts.items():
        print(f"y{y}: {n}")

    tp_per_subj = connectome_index.groupby("RID")["Y"].nunique()
    print("\n#timepoints per subject:")
    for k, v in tp_per_subj.value_counts().sort_index().items():
        print(f"{k} timepoints: {v} subjects")

elif COHORT_NAME == "ADRC":
    unique_subjects = connectome_index["match_id"].dropna().nunique()
    print(f"Unique subjects (match_id): {unique_subjects}")

elif COHORT_NAME == "AD_DECODE":
    unique_subjects = connectome_index["CONN_KEY"].nunique()
    print(f"Unique subjects: {unique_subjects}")

# =========================
# LOAD METADATA
# =========================
print_header(f"{COHORT_NAME} METADATA")

if cfg["metadata_sheet"] is None:
    df_metadata = pd.read_excel(cfg["metadata_path"])
else:
    df_metadata = pd.read_excel(cfg["metadata_path"], sheet_name=cfg["metadata_sheet"])

print(f"Metadata loaded: {df_metadata.shape[0]} rows")

# =========================
# MATCH CONNECTOMES + METADATA
# =========================
print_subheader("MATCHING CONNECTOMES WITH METADATA")

if COHORT_NAME == "HABS":
    df_metadata = df_metadata.copy()
    df_metadata_clean = df_metadata.dropna(subset=[cfg["runno_col"]]).copy()
    df_metadata_clean["runno_fixed"] = df_metadata_clean[cfg["runno_col"]].astype(str).str.strip()

    available_runno = set(connectome_index["CONN_KEY"])
    matched_metadata = df_metadata_clean[df_metadata_clean["runno_fixed"].isin(available_runno)].copy()

    print(f"Matched sessions (metadata & connectome): {len(matched_metadata)} out of {len(available_runno)}")
    df_analysis = matched_metadata.drop_duplicates(subset=[cfg["subject_col"]]).copy()

elif COHORT_NAME == "ADNI":
    df_metadata = df_metadata.copy()

    if cfg["rid_col"] not in df_metadata.columns:
        raise KeyError(f"ADNI metadata must contain {cfg['rid_col']}")

    df_metadata["RID_num"] = pd.to_numeric(df_metadata[cfg["rid_col"]], errors="coerce")
    df_metadata = df_metadata.dropna(subset=["RID_num"]).copy()
    df_metadata["RID_num"] = df_metadata["RID_num"].astype(int)
    df_metadata["RID_4"] = df_metadata["RID_num"].astype(str).str.zfill(4)

    connectome_index["RID_4"] = connectome_index["RID"].astype(int).astype(str).str.zfill(4)

    conn_rids = set(connectome_index["RID_4"])
    meta_rids = set(df_metadata["RID_4"])

    matched_rids = conn_rids & meta_rids
    print(f"Matched subjects (metadata & connectome): {len(matched_rids)} out of {len(conn_rids)}")

    df_analysis_long = df_metadata[df_metadata["RID_4"].isin(matched_rids)].copy()

    if cfg["visit_col"] in df_analysis_long.columns:
        vis = df_analysis_long[cfg["visit_col"]].astype(str).str.strip().str.lower()
        df_analysis_long["_visit_priority"] = np.where(vis.isin(["bl", "m00", "sc", "init"]), 0, 1)
    else:
        df_analysis_long["_visit_priority"] = 1

    def first_nonnull(series):
        s = series.dropna()
        if len(s) == 0:
            return np.nan
        return s.iloc[0]

    rows = []
    for rid4, g in df_analysis_long.groupby("RID_4", sort=True):
        g = g.sort_values("_visit_priority").copy()

        row = {"RID_4": rid4}

        if "RID" in g.columns:
            row["RID"] = first_nonnull(g["RID"])

        row[cfg["age_col"]] = first_nonnull(g[cfg["age_col"]]) if cfg["age_col"] in g.columns else np.nan
        row[cfg["sex_col"]] = first_nonnull(g[cfg["sex_col"]]) if cfg["sex_col"] in g.columns else np.nan
        row[cfg["dx_col"]] = first_nonnull(g[cfg["dx_col"]]) if cfg["dx_col"] in g.columns else np.nan
        row[cfg["apoe_col1"]] = first_nonnull(g[cfg["apoe_col1"]]) if cfg["apoe_col1"] in g.columns else np.nan
        row[cfg["apoe_col2"]] = first_nonnull(g[cfg["apoe_col2"]]) if cfg["apoe_col2"] in g.columns else np.nan

        rows.append(row)

    df_analysis = pd.DataFrame(rows)

elif COHORT_NAME == "ADRC":
    df_metadata = df_metadata.copy()

    df_metadata["ID"] = (
        df_metadata[cfg["metadata_id_col"]]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    df_metadata["match_id"] = df_metadata["ID"].apply(extract_4digit_match_id)

    conn_ids = set(connectome_index["match_id"].dropna())
    meta_ids = set(df_metadata["match_id"].dropna())
    matched_ids = conn_ids & meta_ids

    print(f"Matched subjects (metadata & connectome): {len(matched_ids)} out of {len(conn_ids)}")

    conn_no_meta = sorted(list(conn_ids.difference(meta_ids)))
    meta_no_conn = sorted(list(meta_ids.difference(conn_ids)))

    print(f"Subjects with connectome but NO metadata: {len(conn_no_meta)}")
    if len(conn_no_meta) > 0:
        print("  Example (up to 15):", conn_no_meta[:15])

    print(f"Subjects with metadata but NO connectome: {len(meta_no_conn)}")
    if len(meta_no_conn) > 0:
        print("  Example (up to 15):", meta_no_conn[:15])

    df_analysis = df_metadata[df_metadata["match_id"].isin(matched_ids)].copy()

elif COHORT_NAME == "AD_DECODE":
    df_metadata = df_metadata.copy()

    df_metadata["MRI_Exam_fixed"] = (
        df_metadata[cfg["metadata_mri_exam_col"]]
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    df_metadata_clean = df_metadata.dropna(how="all").copy()
    df_metadata_clean = df_metadata_clean.dropna(subset=[cfg["metadata_mri_exam_col"]]).copy()

    matched_metadata = df_metadata_clean[
        df_metadata_clean["MRI_Exam_fixed"].isin(set(connectome_index["CONN_KEY"]))
    ].copy()

    print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(connectome_index)}")

    df_analysis = matched_metadata.copy()

    pca_cols_found = [c for c in df_analysis.columns if "pca" in c.lower()]
    print("PCA columns found in metadata:", pca_cols_found[:20])
    print(f"Subjects with metadata + connectome: {df_analysis.shape[0]}")

else:
    raise ValueError("Unsupported cohort")

print(f"Analysis dataframe rows: {df_analysis.shape[0]}")

# =========================
# COLUMN SELECTION
# =========================
if COHORT_NAME == "ADNI":
    age_col = cfg["age_col"]
    sex_col = cfg["sex_col"]
    dx_col = cfg["dx_col"]
else:
    age_col = cfg.get("age_col")
    sex_col = cfg.get("sex_col")
    dx_col = cfg.get("dx_col", None)

print("\nUsing columns:")
print("  AGE:", age_col if age_col is not None else "derived")
print("  SEX:", sex_col if sex_col is not None else "derived")
print("  DX :", dx_col if dx_col is not None else "derived")
if COHORT_NAME == "ADNI":
    print("  APOE:", cfg["apoe_col1"], "+", cfg["apoe_col2"])
else:
    print("  APOE:", cfg.get("apoe_col", "derived"))

# =========================
# AGE
# =========================
if age_col in df_analysis.columns:
    age = pd.to_numeric(df_analysis[age_col], errors="coerce")
else:
    age = pd.Series(dtype=float)

if len(age.dropna()) > 0:
    age_mean = age.mean()
    age_std = age.std()
    age_min = age.min()
    age_max = age.max()
    age_mean_sd_str = f"{age_mean:.2f} ± {age_std:.2f}"
    age_range_str = f"[{age_min:.1f}, {age_max:.1f}]"
else:
    age_mean_sd_str = "NA"
    age_range_str = "NA"

# =========================
# SEX
# =========================
if sex_col in df_analysis.columns:
    if COHORT_NAME == "HABS":
        df_analysis["Sex_Label"] = df_analysis[sex_col].map({
            "F": "Female (F)",
            "M": "Male (M)"
        }).fillna(df_analysis[sex_col].astype(str))

    elif COHORT_NAME == "ADRC":
        df_analysis["Sex_Label"] = df_analysis[sex_col].map({
            1: "Male (M)",
            2: "Female (F)",
            "1": "Male (M)",
            "2": "Female (F)"
        }).fillna(df_analysis[sex_col].astype(str))

    else:
        df_analysis["Sex_Label"] = normalize_sex_generic(df_analysis[sex_col])
else:
    df_analysis["Sex_Label"] = np.nan

# =========================
# DIAGNOSIS
# =========================
if COHORT_NAME == "HABS":
    dx_map = {0: "CN", 1: "MCI", 2: "AD/Dementia", 9: "Other/Unknown"}
    df_analysis["DX_Label"] = df_analysis[cfg["dx_col"]].map(dx_map).fillna("Other/Unknown")

elif COHORT_NAME == "ADNI":
    df_analysis["DX_Label"] = df_analysis[dx_col].astype(str).str.strip()
    df_analysis.loc[df_analysis["DX_Label"].isin(["nan", "None", ""]), "DX_Label"] = np.nan

elif COHORT_NAME == "ADRC":
    for c in [cfg["dx_flag_norm"], cfg["dx_flag_mci"], cfg["dx_flag_dem"]]:
        if c not in df_analysis.columns:
            df_analysis[c] = np.nan

    df_analysis["DX_Label"] = "Unknown"
    df_analysis.loc[pd.to_numeric(df_analysis[cfg["dx_flag_norm"]], errors="coerce") == 1, "DX_Label"] = "Normal"
    df_analysis.loc[pd.to_numeric(df_analysis[cfg["dx_flag_mci"]], errors="coerce") == 1, "DX_Label"] = "MCI"
    df_analysis.loc[pd.to_numeric(df_analysis[cfg["dx_flag_dem"]], errors="coerce") == 1, "DX_Label"] = "Demented"

elif COHORT_NAME == "AD_DECODE":
    df_analysis["DX_Label"] = df_analysis[cfg["dx_col"]].astype(str).str.strip()
    df_analysis["DX_Label"] = (
        df_analysis["DX_Label"]
        .fillna("NoRisk")
        .replace(r"^\s*$", "NoRisk", regex=True)
    )
    df_analysis.loc[df_analysis["DX_Label"].isin(["nan", "None", ""]), "DX_Label"] = np.nan

else:
    df_analysis["DX_Label"] = np.nan

# =========================
# APOE
# =========================
if COHORT_NAME == "ADNI":
    c1 = cfg["apoe_col1"]
    c2 = cfg["apoe_col2"]

    if c1 not in df_analysis.columns:
        c1 = next((c for c in df_analysis.columns if "apoe" in c.lower() and "1" in c.lower()), None)
    if c2 not in df_analysis.columns:
        c2 = next((c for c in df_analysis.columns if "apoe" in c.lower() and "2" in c.lower()), None)

    if c1 is not None and c2 is not None:
        df_analysis["APOE_Label"] = normalize_apoe_from_two_cols(df_analysis[c1], df_analysis[c2])
    else:
        print("WARNING: APOE columns not found.")
        df_analysis["APOE_Label"] = np.nan

else:
    apoe_col = cfg.get("apoe_col")
    if apoe_col in df_analysis.columns:
        df_analysis["APOE_Label"] = df_analysis[apoe_col].apply(normalize_apoe_single)
    else:
        df_analysis["APOE_Label"] = np.nan

# =========================
# COUNTS
# =========================
n_total = len(df_analysis)

dx_counts, dx_perc = counts_perc(df_analysis["DX_Label"], denom=n_total)
sex_counts, sex_perc = counts_perc(df_analysis["Sex_Label"], denom=n_total)
apoe_counts, apoe_perc = counts_perc(df_analysis["APOE_Label"], denom=n_total)

# =========================
# PRINT SUMMARY
# =========================
print_header("SUMMARY")

print("=== AGE ===")
print(f"Mean ± SD : {age_mean_sd_str}")
print(f"Range     : {age_range_str}")

print("\n=== DIAGNOSTIC GROUP ===")
if len(dx_counts) == 0:
    print("NA")
else:
    for grp, n in dx_counts.items():
        print(f"{grp:<15}: {n:3d} ({dx_perc[grp]}%)")

print("\n=== SEX ===")
if len(sex_counts) == 0:
    print("NA")
else:
    for sx, n in sex_counts.items():
        print(f"{sx:<15}: {n:3d} ({sex_perc[sx]}%)")

print("\n=== APOE GENOTYPE ===")
if len(apoe_counts) == 0:
    print("NA")
else:
    for gt, n in apoe_counts.items():
        print(f"{gt:<15}: {n:3d} ({apoe_perc[gt]}%)")

# =========================
# SUMMARY TABLE
# =========================
rows = [
    ["Age", "Mean ± SD", age_mean_sd_str],
    ["Age", "Range", age_range_str],
]
rows += [["Diagnostic group", g, f"{dx_counts[g]} ({dx_perc[g]}%)"] for g in dx_counts.index]
rows += [["Sex", s, f"{sex_counts[s]} ({sex_perc[s]}%)"] for s in sex_counts.index]
rows += [["APOE genotype", a, f"{apoe_counts[a]} ({apoe_perc[a]}%)"] for a in apoe_counts.index]

df_summary = pd.DataFrame(rows, columns=["Category", "Value", "Count (%)"])

print(f"\n--- SUMMARY TABLE ({COHORT_NAME}) ---")
print(df_summary)

# =========================
# OPTIONAL SAVES
# =========================
if SAVE_OUTPUTS:
    prefix = COHORT_NAME.replace(" ", "_")

    save_if_requested(df_summary, f"{prefix}_summary_table.csv")

    if COHORT_NAME == "HABS" and cfg["subject_col"] in df_analysis.columns:
        save_if_requested(
            df_analysis[[cfg["subject_col"]]].drop_duplicates(),
            f"{prefix}_matched_subjects.csv"
        )

    elif COHORT_NAME == "ADNI":
        cols_to_save = [c for c in ["RID", "RID_4"] if c in df_analysis.columns]
        if len(cols_to_save) > 0:
            save_if_requested(
                df_analysis[cols_to_save].drop_duplicates(),
                f"{prefix}_matched_subjects.csv"
            )

    elif COHORT_NAME == "ADRC":
        cols_to_save = [c for c in ["ID", "match_id"] if c in df_analysis.columns]
        if len(cols_to_save) > 0:
            save_if_requested(
                df_analysis[cols_to_save].drop_duplicates(),
                f"{prefix}_matched_subjects.csv"
            )

    elif COHORT_NAME == "AD_DECODE" and "MRI_Exam_fixed" in df_analysis.columns:
        save_if_requested(
            df_analysis[["MRI_Exam_fixed"]].drop_duplicates(),
            f"{prefix}_matched_subjects.csv"
        )

print("\nDone.")