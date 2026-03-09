#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PATHS
# =========================================================

WORK = os.environ["WORK"]

SHAP_DIR = os.path.join(WORK, "ines/results/Shap_edges/edges_addecode")
PLOTS_DIR = os.path.join(WORK, "ines/results/Shap_edges/plots_addecode")
PERSONALISED_DIR = os.path.join(PLOTS_DIR, "personalised")
BEESWARM_DIR = os.path.join(PLOTS_DIR, "beeswarm")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PERSONALISED_DIR, exist_ok=True)
os.makedirs(BEESWARM_DIR, exist_ok=True)

print(f"Reading SHAP CSVs from: {SHAP_DIR}")
print(f"Plots will be saved in: {PLOTS_DIR}")

# =========================================================
# REGION NAMES
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
# OPTIONAL: LOAD AGES FROM METADATA
# =========================================================

metadata_path = os.path.join(WORK, "ines/data/AD_DECODE_data4.xlsx")
subject_to_age = {}

if os.path.exists(metadata_path):
    df_meta = pd.read_excel(metadata_path)
    df_meta["MRI_Exam_fixed"] = (
        df_meta["MRI_Exam"]
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    subject_to_age = dict(zip(df_meta["MRI_Exam_fixed"], df_meta["age"]))
    print(f"Loaded ages from metadata for {len(subject_to_age)} subjects.")
else:
    print("Metadata file not found. Subject ages will not be shown automatically.")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_edge_label(node_i, node_j):
    return f"{region_names[int(node_i)]} ↔ {region_names[int(node_j)]}"

def load_subject_shap_csv(subject_id, shap_dir=SHAP_DIR):
    path_csv = os.path.join(shap_dir, f"edge_shap_subject_{subject_id}.csv")
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Could not find file: {path_csv}")

    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()

    if "SHAP_val" not in df.columns:
        raise ValueError(f"'SHAP_val' column not found in {path_csv}")

    if "abs_SHAP" not in df.columns:
        df["abs_SHAP"] = df["SHAP_val"].abs()

    return df

# =========================================================
# PERSONAL SUBJECT PLOT
# =========================================================

def plot_top_edges_addecode(subject_id, subject_age=None, top_n=15, save=True):
    df = load_subject_shap_csv(subject_id)

    top_edges = df.sort_values("abs_SHAP", ascending=False).head(top_n).copy()
    top_edges["Edge"] = top_edges.apply(
        lambda row: get_edge_label(row["Node_i"], row["Node_j"]), axis=1
    )

    shap_vals = top_edges["SHAP_val"].values
    labels = top_edges["Edge"].tolist()
    colors = ["steelblue" if x > 0 else "crimson" for x in shap_vals]

    if subject_age is None:
        subject_age = subject_to_age.get(str(subject_id), None)

    plt.figure(figsize=(10, 6))
    plt.barh(labels, shap_vals, color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP value (edge contribution to predicted age)")

    if subject_age is not None and not pd.isna(subject_age):
        plt.title(f"Top {top_n} SHAP Edges — {subject_id} (Age: {int(subject_age)})")
    else:
        plt.title(f"Top {top_n} SHAP Edges — {subject_id}")

    plt.tight_layout()
    plt.gca().invert_yaxis()

    if save:
        out_path = os.path.join(PERSONALISED_DIR, f"SHAP_edges_{subject_id}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved personalised plot: {out_path}")
    else:
        plt.show()

# =========================================================
# LOAD ALL SHAP CSVs
# =========================================================

def load_all_shap_csvs(shap_dir=SHAP_DIR):
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
    plt.title(f"Top {top_n} Most Important DTI Edges (AD-DECODE)")
    plt.xlabel("SHAP Value")
    plt.ylabel("Edge")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        out_path = os.path.join(BEESWARM_DIR, f"top{top_n}_dti_edges_beeswarm_addecode.png")
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
    plt.title(f"Top {top_n} Edges by Mean |SHAP| (AD-DECODE)")
    plt.tight_layout()

    if save:
        out_path = os.path.join(BEESWARM_DIR, f"top{top_n}_edges_mean_abs_shap_barplot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved barplot: {out_path}")
    else:
        plt.show()

# =========================================================
# AUTO-SELECT YOUNG / MIDDLE / OLD SUBJECTS
# =========================================================

def plot_auto_selected_subjects(top_n=15):
    if not subject_to_age:
        print("No subject ages available. Please call plot_top_edges_addecode() with subject IDs manually.")
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

    plot_top_edges_addecode(subject_young, top_n=top_n, save=True)
    plot_top_edges_addecode(subject_middle, top_n=top_n, save=True)
    plot_top_edges_addecode(subject_old, top_n=top_n, save=True)

# =========================================================
# RUN EXAMPLES
# =========================================================

# 1) Three manual subjects
plot_top_edges_addecode("02231", top_n=15)
plot_top_edges_addecode("02473", top_n=15)
plot_top_edges_addecode("02967", top_n=15)

# 2) Global plots
plot_global_top_edges_beeswarm(top_n=10)
plot_global_top_edges_barplot(top_n=20)

# 3) Optional automatic selection by age
plot_auto_selected_subjects(top_n=15)

print("Done.")