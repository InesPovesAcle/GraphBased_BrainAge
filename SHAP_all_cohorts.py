#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized SHAP edge plots for harmonized cohorts
==================================================

Based on your original shap_plots.py, but generalized so it works for:
    - ADNI
    - ADRC
    - HABS
    - AD_DECODE

What it does
------------
1) Reads edge-level SHAP CSVs from a cohort-specific folder:
       edge_shap_subject_<SUBJECT>.csv
2) Uses a common 84-region atlas name list
3) Creates:
       - personalised per-subject top-edge barplots
       - global beeswarm/stripplot for top edges
       - global mean |SHAP| barplot
       - optional signed mean SHAP barplot
4) Optionally loads ages from cohort-specific metadata / predictions
   so titles can include age and auto-selected young/middle/old subjects

Notes
-----
- This expects edge-SHAP CSVs already computed for each cohort.
- So this script is ONLY for plotting existing SHAP edge outputs.
- It does not compute SHAP itself.
- It preserves the spirit of your original ADDECODE plotting code.

Change only:
    COHORT = "HABS"
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

WORK = os.environ["WORK"]

# =========================================================
# USER CONFIG
# =========================================================

COHORT = "AD_DECODE"   # "ADNI", "ADRC", "HABS", "AD_DECODE"
TOP_N_GLOBAL_BEESWARM = 10
TOP_N_GLOBAL_BARPLOT = 20
TOP_N_SUBJECT = 15
AUTO_SELECT_BY_AGE = True
MAKE_SIGNED_MEAN_BARPLOT = True

# =========================================================
# COHORT PATHS
# =========================================================

COHORT_CONFIG = {
    "ADNI": {
        "shap_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_adni"),
        "plots_dir": os.path.join(WORK, "ines/results/Shap_edges/plots_adni"),
        "pred_csv": os.path.join(WORK, "ines/results/BrainAgePredictionADNI/adni_full_cohort_predictions.csv"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/ADNI/ADNI_metadata.csv"),
        "subject_examples": [],
    },
    "ADRC": {
        "shap_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_adrc"),
        "plots_dir": os.path.join(WORK, "ines/results/Shap_edges/plots_adrc"),
        "pred_csv": os.path.join(WORK, "ines/results/BrainAgePredictionADRC/adrc_full_cohort_predictions.csv"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/ADRC/ADRC_metadata.csv"),
        "subject_examples": [],
    },
    "HABS": {
        "shap_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_habs"),
        "plots_dir": os.path.join(WORK, "ines/results/Shap_edges/plots_habs"),
        "pred_csv": os.path.join(WORK, "ines/results/BrainAgePredictionHABS/habs_full_cohort_predictions.csv"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/HABS/HABS_metadata.csv"),
        "subject_examples": [],
    },
    "AD_DECODE": {
        "shap_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_addecode"),
        "plots_dir": os.path.join(WORK, "ines/results/Shap_edges/plots_addecode"),
        "pred_csv": os.path.join(WORK, "ines/results/BrainAgePredictionADDECODE/addecode_full_cohort_predictions.csv"),
        "metadata_path": os.path.join(WORK, "ines/data/AD_DECODE_data4.xlsx"),
        "subject_examples": ["02231", "02473", "02967"],
    },
}

if COHORT not in COHORT_CONFIG:
    raise ValueError(f"Invalid COHORT: {COHORT}. Choose from {list(COHORT_CONFIG.keys())}")

SHAP_DIR = COHORT_CONFIG[COHORT]["shap_dir"]
PLOTS_DIR = COHORT_CONFIG[COHORT]["plots_dir"]
PRED_CSV = COHORT_CONFIG[COHORT]["pred_csv"]
METADATA_PATH = COHORT_CONFIG[COHORT]["metadata_path"]
SUBJECT_EXAMPLES = COHORT_CONFIG[COHORT]["subject_examples"]

PERSONALISED_DIR = os.path.join(PLOTS_DIR, "personalised")
BEESWARM_DIR = os.path.join(PLOTS_DIR, "beeswarm")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PERSONALISED_DIR, exist_ok=True)
os.makedirs(BEESWARM_DIR, exist_ok=True)

print(f"Cohort: {COHORT}")
print(f"Reading SHAP CSVs from: {SHAP_DIR}")
print(f"Plots will be saved in: {PLOTS_DIR}")
print(f"Pred CSV: {PRED_CSV}")
print(f"Metadata path: {METADATA_PATH}")

# =========================================================
# REGION NAMES (same atlas ordering as your old code)
# =========================================================

region_names = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

if len(region_names) != 84:
    raise ValueError(f"Expected 84 region names, got {len(region_names)}")

# =========================================================
# HELPERS
# =========================================================

def normalize_subject_id(x):
    s = str(x).strip().upper()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"_MASTER_T$", "", s)
    s = re.sub(r"_MASTER$", "", s)
    s = re.sub(r"_T$", "", s)
    s = re.sub(r"_Y(\d+)$", lambda m: f"_y{m.group(1)}", s)

    m = re.fullmatch(r"S0*([0-9]+)", s)
    if m:
        return str(int(m.group(1))).zfill(5)

    m = re.fullmatch(r"D([0-9]+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    m = re.fullmatch(r"ADRC0*([0-9]+)", s)
    if m:
        return m.group(1)[-4:].zfill(4)

    if COHORT == "ADRC" and re.fullmatch(r"[0-9]+", s):
        return s[-4:].zfill(4)

    return s


def load_table_auto(path):
    if not path or not os.path.exists(path):
        return None
    lower = path.lower()
    if lower.endswith(".csv"):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep="\t")
    if lower.endswith(".txt") or lower.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    return None


def normalize_prediction_columns(df):
    rename_map = {}
    aliases = {
        "graph_id": ["graph_id", "Subject_ID", "subject_id", "connectome_key", "ID"],
        "age_true": ["age_true", "Real_Age", "AGE", "Age", "age"],
    }
    for target, names in aliases.items():
        if target in df.columns:
            continue
        for old in names:
            if old in df.columns:
                rename_map[old] = target
                break
    return df.rename(columns=rename_map).copy()


def build_subject_to_age():
    subject_to_age = {}

    pred_df = load_table_auto(PRED_CSV)
    if pred_df is not None:
        pred_df = normalize_prediction_columns(pred_df)
        if "graph_id" in pred_df.columns and "age_true" in pred_df.columns:
            tmp = pred_df[["graph_id", "age_true"]].copy()
            tmp["merge_id"] = tmp["graph_id"].map(normalize_subject_id)
            for _, row in tmp.iterrows():
                if pd.notna(row["age_true"]):
                    subject_to_age[str(row["merge_id"])] = float(row["age_true"])

    meta_df = load_table_auto(METADATA_PATH)
    if meta_df is not None and len(subject_to_age) == 0:
        meta = meta_df.copy()

        candidate_id_cols = [
            "MRI_Exam", "MRI_Exam_fixed", "connectome_key", "Subject ID", "Subject_ID", "graph_id", "ID"
        ]
        candidate_age_cols = ["age", "Age", "AGE", "Real_Age"]

        id_col = next((c for c in candidate_id_cols if c in meta.columns), None)
        age_col = next((c for c in candidate_age_cols if c in meta.columns), None)

        if id_col is not None and age_col is not None:
            meta = meta[[id_col, age_col]].copy()
            meta["merge_id"] = meta[id_col].map(normalize_subject_id)
            for _, row in meta.iterrows():
                if pd.notna(row[age_col]):
                    subject_to_age[str(row["merge_id"])] = float(row[age_col])

    print(f"Loaded ages for {len(subject_to_age)} subjects")
    return subject_to_age


def get_edge_label(node_i, node_j):
    return f"{region_names[int(node_i)]} ↔ {region_names[int(node_j)]}"


def load_subject_shap_csv(subject_id, shap_dir=SHAP_DIR):
    subject_id = str(subject_id)
    path_csv = os.path.join(shap_dir, f"edge_shap_subject_{subject_id}.csv")
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Could not find file: {path_csv}")

    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()

    required = {"Node_i", "Node_j", "SHAP_val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path_csv}")

    if "abs_SHAP" not in df.columns:
        df["abs_SHAP"] = df["SHAP_val"].abs()

    return df


def load_all_shap_csvs(shap_dir=SHAP_DIR):
    if not os.path.exists(shap_dir):
        raise FileNotFoundError(f"SHAP directory not found: {shap_dir}")

    shap_dfs = []
    for fname in os.listdir(shap_dir):
        if fname.endswith(".csv") and fname.startswith("edge_shap_subject_"):
            fpath = os.path.join(shap_dir, fname)
            df = pd.read_csv(fpath)
            df.columns = df.columns.str.strip()
            df["subject"] = fname.replace("edge_shap_subject_", "").replace(".csv", "")
            if "abs_SHAP" not in df.columns:
                df["abs_SHAP"] = df["SHAP_val"].abs()
            shap_dfs.append(df)

    if len(shap_dfs) == 0:
        raise RuntimeError(f"No edge_shap_subject_*.csv files found in {shap_dir}")

    df_all = pd.concat(shap_dfs, ignore_index=True)
    return df_all

# =========================================================
# PERSONAL SUBJECT PLOT
# =========================================================

def plot_top_edges_subject(subject_id, subject_to_age, top_n=15, save=True):
    df = load_subject_shap_csv(subject_id)

    top_edges = df.sort_values("abs_SHAP", ascending=False).head(top_n).copy()
    top_edges["Edge"] = top_edges.apply(
        lambda row: get_edge_label(row["Node_i"], row["Node_j"]), axis=1
    )

    shap_vals = top_edges["SHAP_val"].values
    labels = top_edges["Edge"].tolist()
    colors = ["steelblue" if x > 0 else "crimson" for x in shap_vals]

    subject_age = subject_to_age.get(str(subject_id), None)

    plt.figure(figsize=(10, 6))
    plt.barh(labels, shap_vals, color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP value (edge contribution to model output)")

    if subject_age is not None and not pd.isna(subject_age):
        plt.title(f"Top {top_n} SHAP Edges — {COHORT} — {subject_id} (Age: {int(subject_age)})")
    else:
        plt.title(f"Top {top_n} SHAP Edges — {COHORT} — {subject_id}")

    plt.tight_layout()
    plt.gca().invert_yaxis()

    if save:
        out_path = os.path.join(PERSONALISED_DIR, f"SHAP_edges_{COHORT}_{subject_id}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved personalised plot: {out_path}")
    else:
        plt.show()

# =========================================================
# GLOBAL STRIPPLOT / "BEESWARM"
# =========================================================

def plot_global_top_edges_beeswarm(top_n=10, save=True):
    df_all = load_all_shap_csvs()

    mean_shap = (
        df_all.groupby(["Node_i", "Node_j"])["abs_SHAP"]
        .mean()
        .reset_index()
    )

    top_edges = mean_shap.sort_values("abs_SHAP", ascending=False).head(top_n).copy()
    top_pairs = set(zip(top_edges["Node_i"], top_edges["Node_j"]))

    df_top = df_all[
        df_all.apply(lambda row: (row["Node_i"], row["Node_j"]) in top_pairs, axis=1)
    ].copy()

    df_top["Edge"] = df_top.apply(
        lambda row: get_edge_label(row["Node_i"], row["Node_j"]), axis=1
    )

    edge_order = [
        get_edge_label(i, j)
        for i, j in zip(top_edges["Node_i"], top_edges["Node_j"])
    ]

    plt.figure(figsize=(11, 7))
    sns.stripplot(
        data=df_top,
        x="SHAP_val",
        y="Edge",
        order=edge_order,
        jitter=True,
        alpha=0.6,
        size=4
    )
    plt.title(f"Top {top_n} Most Important Edges ({COHORT})")
    plt.xlabel("SHAP Value")
    plt.ylabel("Edge")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        out_path = os.path.join(BEESWARM_DIR, f"top{top_n}_dti_edges_beeswarm_{COHORT.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved beeswarm plot: {out_path}")
    else:
        plt.show()

# =========================================================
# GLOBAL MEAN |SHAP| BARPLOT
# =========================================================

def plot_global_top_edges_barplot(top_n=20, save=True):
    df_all = load_all_shap_csvs()

    mean_shap = (
        df_all.groupby(["Node_i", "Node_j"])["abs_SHAP"]
        .mean()
        .reset_index()
        .sort_values("abs_SHAP", ascending=False)
        .head(top_n)
        .copy()
    )

    mean_shap["Edge"] = mean_shap.apply(
        lambda row: get_edge_label(row["Node_i"], row["Node_j"]), axis=1
    )

    mean_shap = mean_shap.iloc[::-1]

    plt.figure(figsize=(11, 8))
    plt.barh(mean_shap["Edge"], mean_shap["abs_SHAP"])
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Edge")
    plt.title(f"Top {top_n} Edges by Mean |SHAP| ({COHORT})")
    plt.tight_layout()

    if save:
        out_path = os.path.join(BEESWARM_DIR, f"top{top_n}_edges_mean_abs_shap_barplot_{COHORT.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved barplot: {out_path}")
    else:
        plt.show()

# =========================================================
# GLOBAL SIGNED MEAN SHAP BARPLOT
# =========================================================

def plot_global_top_edges_signed_barplot(top_n=20, save=True):
    df_all = load_all_shap_csvs()

    mean_signed = (
        df_all.groupby(["Node_i", "Node_j"])["SHAP_val"]
        .mean()
        .reset_index()
    )
    mean_abs = (
        df_all.groupby(["Node_i", "Node_j"])["abs_SHAP"]
        .mean()
        .reset_index()
        .rename(columns={"abs_SHAP": "mean_abs_SHAP"})
    )
    merged = mean_signed.merge(mean_abs, on=["Node_i", "Node_j"], how="inner")
    merged = merged.sort_values("mean_abs_SHAP", ascending=False).head(top_n).copy()

    merged["Edge"] = merged.apply(
        lambda row: get_edge_label(row["Node_i"], row["Node_j"]), axis=1
    )
    merged = merged.iloc[::-1]
    colors = ["steelblue" if x > 0 else "crimson" for x in merged["SHAP_val"]]

    plt.figure(figsize=(11, 8))
    plt.barh(merged["Edge"], merged["SHAP_val"], color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Mean SHAP")
    plt.ylabel("Edge")
    plt.title(f"Top {top_n} Edges by signed mean SHAP ({COHORT})")
    plt.tight_layout()

    if save:
        out_path = os.path.join(BEESWARM_DIR, f"top{top_n}_edges_signed_mean_shap_barplot_{COHORT.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved signed barplot: {out_path}")
    else:
        plt.show()

# =========================================================
# AUTO-SELECT YOUNG / MIDDLE / OLD SUBJECTS
# =========================================================

def plot_auto_selected_subjects(subject_to_age, top_n=15):
    if not subject_to_age:
        print("No subject ages available. Skipping auto selection.")
        return

    available_subjects = []
    for fname in os.listdir(SHAP_DIR):
        if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
            sid = fname.replace("edge_shap_subject_", "").replace(".csv", "")
            if sid in subject_to_age and pd.notna(subject_to_age[sid]):
                available_subjects.append((sid, subject_to_age[sid]))

    if len(available_subjects) < 3:
        print("Not enough subjects with age and SHAP files to auto-select.")
        return

    df_age = pd.DataFrame(available_subjects, columns=["Subject_ID", "Age"])
    subject_young = df_age.loc[df_age["Age"].idxmin(), "Subject_ID"]
    subject_old = df_age.loc[df_age["Age"].idxmax(), "Subject_ID"]
    median_age = df_age["Age"].median()
    subject_middle = df_age.iloc[(df_age["Age"] - median_age).abs().argsort().iloc[0]]["Subject_ID"]

    print("Auto-selected subjects:")
    print("  Young :", subject_young)
    print("  Middle:", subject_middle)
    print("  Old   :", subject_old)

    plot_top_edges_subject(subject_young, subject_to_age, top_n=top_n, save=True)
    plot_top_edges_subject(subject_middle, subject_to_age, top_n=top_n, save=True)
    plot_top_edges_subject(subject_old, subject_to_age, top_n=top_n, save=True)

# =========================================================
# RUN
# =========================================================

subject_to_age = build_subject_to_age()

# manual examples if provided
for sid in SUBJECT_EXAMPLES:
    try:
        plot_top_edges_subject(sid, subject_to_age, top_n=TOP_N_SUBJECT, save=True)
    except Exception as e:
        print(f"Skipping example subject {sid}: {e}")

# global plots
plot_global_top_edges_beeswarm(top_n=TOP_N_GLOBAL_BEESWARM, save=True)
plot_global_top_edges_barplot(top_n=TOP_N_GLOBAL_BARPLOT, save=True)

if MAKE_SIGNED_MEAN_BARPLOT:
    plot_global_top_edges_signed_barplot(top_n=TOP_N_GLOBAL_BARPLOT, save=True)

# optional auto-selected subjects
if AUTO_SELECT_BY_AGE:
    plot_auto_selected_subjects(subject_to_age, top_n=TOP_N_SUBJECT)

print("Done.")
