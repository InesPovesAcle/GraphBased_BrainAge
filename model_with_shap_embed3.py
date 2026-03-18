#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:42:55 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDECODE brain-age training on healthy subjects + inference on all subjects
- Threshold 70
- 7-fold CV
- 10 repeats per fold
- Contrastive pretraining ONCE per OUTER fold
- Reuse pretrained encoder across repeats
- SHAP-guided graph augmentations (protect important edges)
- Supervised brain-age prediction after pretraining
- Proper fold-wise bias correction:
    fit correction on TRAIN predictions only
    apply correction to TEST predictions only
- Subject-level aggregated CV metrics
- Final model trained on all healthy subjects
- Final inference on ALL subjects (NoRisk / Familial / MCI / AD)
- Healthy-derived bias correction applied to all-subject inference
- Save metrics, predictions, plots and final model
"""

################# OUTPUT DIR ################

import os

WORK = os.environ["WORK"]
output_dir = os.path.join(WORK, "ines/results/Model_with_shap_guided_contrastive_learning")
os.makedirs(output_dir, exist_ok=True)

################# IMPORTS ################

import copy
import zipfile
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

################# REPRODUCIBILITY ################

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

################# PATHS ################

zip_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/connectomes/AD_DECODE_connectome_act.zip"
)

metadata_path = os.path.join(
    WORK,
    "ines/data/AD_DECODE_data4.xlsx"
)

pca_path = os.path.join(
    WORK,
    "ines/data/PCA_human_blood_top30.csv"
)

fa_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt"
)

md_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt"
)

vol_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume.txt"
)

shap_embed_path = os.path.join(
    WORK,
    "ines/results/Shap_edges/edges_addecode/edge_shap_matrix_all_subjects.csv"
)

directory_inside_zip = "connectome_act/"

##################### DEVICE #######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

####################### CONNECTOMES ###############################

print("ADDECODE CONNECTOMES\n")

connectomes = {}

with zipfile.ZipFile(zip_path, "r") as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                df = pd.read_csv(f, header=None)
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[subject_id] = df

print(f"Total connectome matrices loaded: {len(connectomes)}")

filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)
    if match:
        num_id = match.group(1).zfill(5)
        cleaned_connectomes[num_id] = v

print()

############################## METADATA ##############################

print("ADDECODE METADATA\n")

df_metadata = pd.read_excel(metadata_path)

df_metadata["MRI_Exam_fixed"] = (
    df_metadata["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

df_metadata_cleaned = df_metadata.dropna(how="all")
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["MRI_Exam"])

print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()

#################### MATCH CONNECTOMES & METADATA ####################

print("MATCHING CONNECTOMES WITH METADATA")

matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}

df_matched_connectomes = matched_metadata.copy()

#################### RISK DISTRIBUTION ####################

if "Risk" in df_matched_connectomes.columns:
    risk_filled = df_matched_connectomes["Risk"].fillna("NoRisk").replace(r"^\s*$", "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()

#################### HEALTHY / ALL SPLITS ####################

print("FILTERING OUT AD AND MCI SUBJECTS FOR TRAINING")

df_matched_addecode_healthy = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
].copy()

df_matched_all = df_matched_connectomes.copy()

print(f"Subjects before removing AD/MCI: {len(df_matched_connectomes)}")
print(f"Subjects after removing AD/MCI: {len(df_matched_addecode_healthy)}")
print()

if "Risk" in df_matched_addecode_healthy.columns:
    risk_filled = df_matched_addecode_healthy["Risk"].fillna("NoRisk").replace(r"^\s*$", "NoRisk", regex=True)
    print("Risk distribution in healthy matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()

matched_connectomes_healthy_addecode = {
    row["MRI_Exam_fixed"]: matched_connectomes[row["MRI_Exam_fixed"]]
    for _, row in df_matched_addecode_healthy.iterrows()
}

matched_connectomes_all = {
    row["MRI_Exam_fixed"]: matched_connectomes[row["MRI_Exam_fixed"]]
    for _, row in df_matched_all.iterrows()
}

print(f"Connectomes selected for healthy training: {len(matched_connectomes_healthy_addecode)}")
print(f"Connectomes selected for all-subject inference: {len(matched_connectomes_all)}")
print()

########################### PCA GENES ###############################

print("PCA GENES")

df_pca = pd.read_csv(pca_path)
df_pca["ID_fixed"] = df_pca["ID"].astype(str).str.upper().str.replace("_", "", regex=False)

df_matched_addecode_healthy["IDRNA_fixed"] = (
    df_matched_addecode_healthy["IDRNA"].astype(str).str.upper().str.replace("_", "", regex=False)
)
df_matched_all["IDRNA_fixed"] = (
    df_matched_all["IDRNA"].astype(str).str.upper().str.replace("_", "", regex=False)
)

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_healthy_withConnectome = df_matched_addecode_healthy.merge(
    df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed"
)

df_metadata_PCA_all_withConnectome = df_matched_all.merge(
    df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed"
)

print(f"Healthy subjects with metadata+connectome: {df_matched_addecode_healthy.shape[0]}")
print(f"Healthy subjects with metadata+PCA+connectome: {df_metadata_PCA_healthy_withConnectome.shape[0]}")
print(f"All subjects with metadata+connectome: {df_matched_all.shape[0]}")
print(f"All subjects with metadata+PCA+connectome: {df_metadata_PCA_all_withConnectome.shape[0]}")
print()

####################### FA MD VOL #############################

df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"].reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col.startswith("S")]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)

df_md = pd.read_csv(md_path, sep="\t")
df_md = df_md[1:]
df_md = df_md[df_md["ROI"] != "0"].reset_index(drop=True)
subject_cols_md = [col for col in df_md.columns if col.startswith("S")]
df_md_transposed = df_md[subject_cols_md].transpose()
df_md_transposed.columns = [f"ROI_{i+1}" for i in range(df_md_transposed.shape[1])]
df_md_transposed.index.name = "subject_id"
df_md_transposed = df_md_transposed.astype(float)

df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"].reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col.startswith("S")]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

def clean_subject_df(df_t):
    cleaned = {}
    for subj in df_t.index:
        match = re.search(r"S(\d{5})", subj)
        if match:
            subj_id = match.group(1)
            if subj_id not in cleaned:
                cleaned[subj_id] = df_t.loc[subj]
    out = pd.DataFrame.from_dict(cleaned, orient="index")
    out.index.name = "subject_id"
    return out

df_fa_transposed_cleaned = clean_subject_df(df_fa_transposed)
df_md_transposed_cleaned = clean_subject_df(df_md_transposed)
df_vol_transposed_cleaned = clean_subject_df(df_vol_transposed)

multimodal_features_dict = {}

for subj_id in df_fa_transposed_cleaned.index:
    if subj_id in df_md_transposed_cleaned.index and subj_id in df_vol_transposed_cleaned.index:
        fa = torch.tensor(df_fa_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        stacked = torch.stack([fa, md, vol], dim=1)
        multimodal_features_dict[subj_id] = stacked

print("Subjects with FA, MD, and Vol features:", len(multimodal_features_dict))

####################### NORMALIZATION HELPERS #############################

def fit_nodewise_normalization(feature_dict, subject_ids):
    stack = torch.stack([feature_dict[sid] for sid in subject_ids if sid in feature_dict])
    means = stack.mean(dim=0)
    stds = stack.std(dim=0) + 1e-8
    return means, stds

def apply_nodewise_normalization(feature_dict, means, stds):
    out = {}
    for sid, feats in feature_dict.items():
        out[sid] = (feats - means) / stds
    return out

def fit_zscore_stats(df, cols):
    stats = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mu = series.mean()
        sd = series.std()
        if pd.isna(sd) or sd == 0:
            sd = 1.0
        stats[col] = (mu, sd)
    return stats

def apply_zscore_stats(df, cols, stats):
    df = df.copy()
    for col in cols:
        mu, sd = stats[col]
        df[col] = (pd.to_numeric(df[col], errors="coerce") - mu) / sd
    return df

####################### THRESHOLD + LOG CONNECTOMES #############################

def threshold_connectome(matrix, percentile=100):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

log_thresholded_connectomes_healthy = {}
for subject, matrix in matched_connectomes_healthy_addecode.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=70)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes_healthy[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)

log_thresholded_connectomes_all = {}
for subject, matrix in matched_connectomes_all.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=70)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes_all[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)

####################### GRAPH METRICS #############################

def compute_nodewise_clustering_coefficients(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]
    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)

def compute_clustering_coefficient(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.average_clustering(G, weight="weight")

def compute_path_length(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        weight = matrix.iloc[u, v]
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        return nx.average_shortest_path_length(G, weight="distance")
    except Exception:
        return float("nan")

addecode_healthy_metadata_pca = df_metadata_PCA_healthy_withConnectome.reset_index(drop=True)
addecode_healthy_metadata_pca["Clustering_Coeff"] = np.nan
addecode_healthy_metadata_pca["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes_healthy.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "Clustering_Coeff"
        ] = clustering
        addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "Path_Length"
        ] = path
    except Exception as e:
        print(f"Failed to compute healthy metrics for subject {subject}: {e}")

addecode_all_metadata_pca = df_metadata_PCA_all_withConnectome.reset_index(drop=True)
addecode_all_metadata_pca["Clustering_Coeff"] = np.nan
addecode_all_metadata_pca["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes_all.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        addecode_all_metadata_pca.loc[
            addecode_all_metadata_pca["MRI_Exam_fixed"] == subject, "Clustering_Coeff"
        ] = clustering
        addecode_all_metadata_pca.loc[
            addecode_all_metadata_pca["MRI_Exam_fixed"] == subject, "Path_Length"
        ] = path
    except Exception as e:
        print(f"Failed to compute all-subject metrics for subject {subject}: {e}")

####################### ENCODE CATEGORICALS + APPLY HEALTHY NORMALIZATION #############################

# Fit encoders on healthy only, apply to all
le_sex = LabelEncoder()
le_sex.fit(addecode_healthy_metadata_pca["sex"].astype(str))
addecode_healthy_metadata_pca["sex_encoded"] = le_sex.transform(addecode_healthy_metadata_pca["sex"].astype(str))
addecode_all_metadata_pca["sex_encoded"] = le_sex.transform(addecode_all_metadata_pca["sex"].astype(str))

le_genotype = LabelEncoder()
le_genotype.fit(addecode_healthy_metadata_pca["genotype"].astype(str))
addecode_healthy_metadata_pca["genotype"] = le_genotype.transform(addecode_healthy_metadata_pca["genotype"].astype(str))
addecode_all_metadata_pca["genotype"] = le_genotype.transform(addecode_all_metadata_pca["genotype"].astype(str))

numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ["PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"]

# Fit healthy-only zscore stats, apply to healthy and all
healthy_stats_num = fit_zscore_stats(addecode_healthy_metadata_pca, numerical_cols)
healthy_stats_pca = fit_zscore_stats(addecode_healthy_metadata_pca, pca_cols)

addecode_healthy_metadata_pca = apply_zscore_stats(addecode_healthy_metadata_pca, numerical_cols, healthy_stats_num)
addecode_healthy_metadata_pca = apply_zscore_stats(addecode_healthy_metadata_pca, pca_cols, healthy_stats_pca)

addecode_all_metadata_pca = apply_zscore_stats(addecode_all_metadata_pca, numerical_cols, healthy_stats_num)
addecode_all_metadata_pca = apply_zscore_stats(addecode_all_metadata_pca, pca_cols, healthy_stats_pca)

# Fit healthy-only node-feature normalization, apply to all
healthy_subjects_for_node_norm = set(addecode_healthy_metadata_pca["MRI_Exam_fixed"]).intersection(multimodal_features_dict.keys())
node_means, node_stds = fit_nodewise_normalization(multimodal_features_dict, healthy_subjects_for_node_norm)
normalized_node_features_dict_all = apply_nodewise_normalization(multimodal_features_dict, node_means, node_stds)

####################### BUILD GLOBAL FEATURE TENSORS #############################

subject_to_demographic_tensor_healthy = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

subject_to_graphmetric_tensor_healthy = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

subject_to_pca_tensor_healthy = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

subject_to_demographic_tensor_all = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_all_metadata_pca.iterrows()
}

subject_to_graphmetric_tensor_all = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_all_metadata_pca.iterrows()
}

subject_to_pca_tensor_all = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_all_metadata_pca.iterrows()
}

####################### LOAD EDGE-LEVEL SHAP #############################

df_shap = pd.read_csv(shap_embed_path)
df_shap["Subject_ID_fixed"] = df_shap["Subject_ID"].astype(str).str.zfill(5)

shap_feature_cols = [c for c in df_shap.columns if c not in ["Subject_ID", "Subject_ID_fixed"]]
expected_num_edges = 84 * 83 // 2

if len(shap_feature_cols) != expected_num_edges:
    raise ValueError(
        f"Your SHAP file has {len(shap_feature_cols)} columns, but SHAP-guided edge protection "
        f"needs {expected_num_edges} edge-level SHAP values (one per upper-triangular edge). "
        f"If your current file contains compressed SHAP embeddings, use the original edge-level SHAP file instead."
    )

subject_to_shap_edge_scores = {
    row["Subject_ID_fixed"]: torch.tensor(
        row[shap_feature_cols].values.astype(np.float32),
        dtype=torch.float
    )
    for _, row in df_shap.iterrows()
}

##################### MATRIX TO GRAPH FUNCTION #######################

def matrix_to_graph(matrix, subject_id, node_features_dict, shap_edge_dict):
    n_nodes = matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)

    upper_edge_weights = matrix.values[upper_i, upper_j]
    upper_shap_scores = shap_edge_dict[subject_id].cpu().numpy()

    src = np.concatenate([upper_i, upper_j])
    dst = np.concatenate([upper_j, upper_i])

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    edge_attr = torch.tensor(
        np.concatenate([upper_edge_weights, upper_edge_weights]),
        dtype=torch.float
    ).unsqueeze(-1)

    shap_edge_scores = torch.tensor(
        np.concatenate([upper_shap_scores, upper_shap_scores]),
        dtype=torch.float
    ).unsqueeze(-1)

    node_feats = node_features_dict[subject_id]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)

    full_node_features = torch.cat([node_feats, clustering_tensor], dim=1)
    node_features = 0.5 * full_node_features

    return edge_index, edge_attr, node_features, shap_edge_scores

################# CONVERT HEALTHY MATRICES TO GRAPH ################

graph_data_list_addecode = []
final_subjects_with_all_data_healthy = []

for subject, matrix_log in log_thresholded_connectomes_healthy.items():
    try:
        if subject not in subject_to_demographic_tensor_healthy:
            continue
        if subject not in subject_to_graphmetric_tensor_healthy:
            continue
        if subject not in subject_to_pca_tensor_healthy:
            continue
        if subject not in normalized_node_features_dict_all:
            continue
        if subject not in subject_to_shap_edge_scores:
            continue

        edge_index, edge_attr, node_features, shap_edge_scores = matrix_to_graph(
            matrix_log,
            subject_id=subject,
            node_features_dict=normalized_node_features_dict_all,
            shap_edge_dict=subject_to_shap_edge_scores
        )

        age_row = addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue

        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        demo_tensor = subject_to_demographic_tensor_healthy[subject]
        graph_tensor = subject_to_graphmetric_tensor_healthy[subject]
        pca_tensor = subject_to_pca_tensor_healthy[subject]
        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0),
            shap_edge_scores=shap_edge_scores
        )
        data.subject_id = subject

        graph_data_list_addecode.append(data)
        final_subjects_with_all_data_healthy.append(subject)

        if len(graph_data_list_addecode) == 1:
            print("\nExample HEALTHY PyTorch Geometric Data object:")
            print("→ Node features shape:", data.x.shape)
            print("→ Edge index shape:", data.edge_index.shape)
            print("→ Edge attr shape:", data.edge_attr.shape)
            print("→ SHAP edge score shape:", data.shap_edge_scores.shape)
            print("→ Global features shape:", data.global_features.shape)
            print("→ Target age (y):", data.y.item())

    except Exception as e:
        print(f"Failed to process healthy subject {subject}: {e}")

torch.save(graph_data_list_addecode, os.path.join(output_dir, "graph_data_list_addecode_healthy.pt"))
print("Saved:", os.path.join(output_dir, "graph_data_list_addecode_healthy.pt"))

print()
expected_healthy = set(subject_to_pca_tensor_healthy.keys())
actual_healthy = set(final_subjects_with_all_data_healthy)
missing_healthy = expected_healthy - actual_healthy

print(f"Healthy subjects with PCA but no graph: {missing_healthy}")
print(f"Total healthy graphs created: {len(actual_healthy)} / Expected: {len(expected_healthy)}")
print()

################# CONVERT ALL MATRICES TO GRAPH ################

graph_data_list_all = []
final_subjects_with_all_data_all = []

risk_map_from_metadata = {
    row["MRI_Exam_fixed"]: row["Risk"] if pd.notna(row["Risk"]) else "NoRisk"
    for _, row in addecode_all_metadata_pca.iterrows()
}

for subject, matrix_log in log_thresholded_connectomes_all.items():
    try:
        if subject not in subject_to_demographic_tensor_all:
            continue
        if subject not in subject_to_graphmetric_tensor_all:
            continue
        if subject not in subject_to_pca_tensor_all:
            continue
        if subject not in normalized_node_features_dict_all:
            continue
        if subject not in subject_to_shap_edge_scores:
            continue

        edge_index, edge_attr, node_features, shap_edge_scores = matrix_to_graph(
            matrix_log,
            subject_id=subject,
            node_features_dict=normalized_node_features_dict_all,
            shap_edge_dict=subject_to_shap_edge_scores
        )

        age_row = addecode_all_metadata_pca.loc[
            addecode_all_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue

        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        demo_tensor = subject_to_demographic_tensor_all[subject]
        graph_tensor = subject_to_graphmetric_tensor_all[subject]
        pca_tensor = subject_to_pca_tensor_all[subject]
        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0),
            shap_edge_scores=shap_edge_scores
        )
        data.subject_id = subject
        data.risk = str(risk_map_from_metadata.get(subject, "Unknown"))

        graph_data_list_all.append(data)
        final_subjects_with_all_data_all.append(subject)

        if len(graph_data_list_all) == 1:
            print("\nExample ALL-SUBJECT PyTorch Geometric Data object:")
            print("→ subject_id:", data.subject_id)
            print("→ risk:", data.risk)
            print("→ Node features shape:", data.x.shape)
            print("→ Global features shape:", data.global_features.shape)
            print("→ Target age (y):", data.y.item())

    except Exception as e:
        print(f"Failed to process all-subject subject {subject}: {e}")

torch.save(graph_data_list_all, os.path.join(output_dir, "graph_data_list_addecode_all_subjects.pt"))
print("Saved:", os.path.join(output_dir, "graph_data_list_addecode_all_subjects.pt"))

print()
expected_all = set(subject_to_pca_tensor_all.keys())
actual_all = set(final_subjects_with_all_data_all)
missing_all = expected_all - actual_all

print(f"All subjects with PCA but no graph: {missing_all}")
print(f"Total all-subject graphs created: {len(actual_all)} / Expected: {len(expected_all)}")
print()

###################### CONTRASTIVE UTILITIES #########################

def clone_batch(data):
    out = copy.copy(data)
    for key, value in data:
        if torch.is_tensor(value):
            out[key] = value.clone()
    return out

def shap_guided_view(
    batch,
    low_drop_prob=0.35,
    high_drop_prob=0.05,
    protect_top_frac=0.20,
    edge_noise_std=0.01
):
    view = clone_batch(batch)

    edge_scores = view.shap_edge_scores.view(-1).abs()
    edge_batch = view.batch[view.edge_index[0]]

    keep_mask = torch.zeros(edge_scores.size(0), dtype=torch.bool, device=edge_scores.device)
    unique_graph_ids = torch.unique(edge_batch)

    for gid in unique_graph_ids:
        idx = torch.where(edge_batch == gid)[0]
        if idx.numel() == 0:
            continue

        local_scores = edge_scores[idx]
        q = max(0.0, min(1.0, 1.0 - protect_top_frac))
        threshold = torch.quantile(local_scores, q)
        is_high_importance = local_scores >= threshold

        keep_probs = torch.where(
            is_high_importance,
            torch.full_like(local_scores, 1.0 - high_drop_prob),
            torch.full_like(local_scores, 1.0 - low_drop_prob)
        )

        sampled_keep = torch.rand_like(keep_probs) < keep_probs

        if sampled_keep.sum() == 0:
            top_idx = torch.argmax(local_scores)
            sampled_keep[top_idx] = True

        keep_mask[idx] = sampled_keep

    view.edge_index = view.edge_index[:, keep_mask]
    view.edge_attr = view.edge_attr[keep_mask]
    view.shap_edge_scores = view.shap_edge_scores[keep_mask]

    noise = torch.randn_like(view.edge_attr) * edge_noise_std
    view.edge_attr = torch.clamp(view.edge_attr + noise, min=0.0)

    return view

def nt_xent_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(representations, representations.t()) / temperature

    batch_size_local = z1.size(0)
    labels = torch.arange(batch_size_local, device=z1.device)
    labels = torch.cat([labels + batch_size_local, labels], dim=0)

    mask = torch.eye(2 * batch_size_local, dtype=torch.bool, device=z1.device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss

###################### MODELS #########################

class GraphEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, graph_emb_dim=128):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.gnn1 = GATv2Conv(hidden_channels, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

        self.post_pool = nn.Sequential(
            nn.Linear(128, graph_emb_dim),
            nn.ReLU()
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.node_embed(x)

        x = self.gnn1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.post_pool(x)

        return x

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, proj_dim)
        )

    def forward(self, data):
        h = self.encoder(data)
        z = self.projector(h)
        return h, z

class BrainAgeRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.pca_head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 16 + 16 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x_graph = self.encoder(data)

        global_feats = data.global_features.to(x_graph.device).squeeze(1)
        meta_embed = self.meta_head(global_feats[:, 0:4])
        graph_embed = self.graph_head(global_feats[:, 4:6])
        pca_embed = self.pca_head(global_feats[:, 6:])

        x = torch.cat([x_graph, meta_embed, graph_embed, pca_embed], dim=1)
        x = self.fc(x)
        return x

###################### TRAINING FUNCTIONS #########################

def train_contrastive_epoch(model, loader, optimizer, temperature=0.2):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        view1 = shap_guided_view(
            batch,
            low_drop_prob=0.25,
            high_drop_prob=0.03,
            protect_top_frac=0.20,
            edge_noise_std=0.005
        ).to(device)

        view2 = shap_guided_view(
            batch,
            low_drop_prob=0.45,
            high_drop_prob=0.05,
            protect_top_frac=0.20,
            edge_noise_std=0.010
        ).to(device)

        optimizer.zero_grad()
        _, z1 = model(view1)
        _, z2 = model(view2)

        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def train_supervised_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data).view(-1)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_supervised(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).view(-1)
            loss = criterion(pred, data.y)
            total_loss += loss.item()

    return total_loss / len(loader)

def get_predictions(model, loader):
    model.eval()

    all_true, all_pred, all_ids = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).view(-1).cpu().numpy()
            true = data.y.cpu().numpy()
            ids = [str(sid) for sid in data.subject_id]

            all_pred.extend(pred.tolist())
            all_true.extend(true.tolist())
            all_ids.extend(ids)

    return np.array(all_true), np.array(all_pred), all_ids

###################### BIAS CORRECTION #########################

def fit_bias_correction(y_true_train, y_pred_train):
    reg = LinearRegression()
    reg.fit(np.array(y_true_train).reshape(-1, 1), np.array(y_pred_train))
    a = float(reg.coef_[0])
    b = float(reg.intercept_)
    return a, b

def apply_bias_correction(y_pred, a, b):
    eps = 1e-8
    if abs(a) < eps:
        return np.array(y_pred).copy()
    return (np.array(y_pred) - b) / a

def compute_bag_slope(y_true, y_pred):
    bag = np.array(y_pred) - np.array(y_true)
    reg = LinearRegression()
    reg.fit(np.array(y_true).reshape(-1, 1), bag)
    return float(reg.coef_[0]), float(reg.intercept_)

###################### SPLIT HELPERS #########################

def make_inner_train_val_split(train_data, val_fraction=0.15, random_state=42):
    if len(train_data) < 10:
        return train_data, train_data

    ages = np.array([float(d.y.item()) for d in train_data])

    try:
        age_bins_local = pd.qcut(ages, q=min(5, len(np.unique(ages))), labels=False, duplicates="drop")
    except Exception:
        age_bins_local = np.zeros(len(ages), dtype=int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_state
    )

    idx = np.arange(len(train_data))
    split = list(splitter.split(idx, age_bins_local))[0]
    subtrain_idx, val_idx = split

    subtrain_data = [train_data[i] for i in subtrain_idx]
    val_data = [train_data[i] for i in val_idx]

    return subtrain_data, val_data

###################### CV SETUP #########################

contrastive_epochs = 80
contrastive_lr = 1e-3

supervised_epochs = 300
patience = 40
k = 7
batch_size = 6
repeats_per_fold = 10

graph_subject_ids = [data.subject_id for data in graph_data_list_addecode]

df_filtered = addecode_healthy_metadata_pca[
    addecode_healthy_metadata_pca["MRI_Exam_fixed"].isin(graph_subject_ids)
].copy()

df_filtered = df_filtered.drop_duplicates(subset="MRI_Exam_fixed", keep="first")
df_filtered = df_filtered.set_index("MRI_Exam_fixed")
df_filtered = df_filtered.loc[df_filtered.index.intersection(graph_subject_ids)]
df_filtered = df_filtered.loc[graph_subject_ids].reset_index()

print("Final matched lengths for HEALTHY CV:")
print(" len(graphs):", len(graph_data_list_addecode))
print(" len(metadata):", len(df_filtered))

ages = df_filtered["age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False, duplicates="drop")

print("Aligned bins:", len(age_bins))

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

###################### STORAGE #########################

all_train_losses = []
all_val_losses = []

all_y_true_raw = []
all_y_pred_raw = []
all_y_pred_bc = []
all_subject_ids = []

fold_mae_raw_list = []
fold_rmse_raw_list = []
fold_r2_raw_list = []

fold_mae_bc_list = []
fold_rmse_bc_list = []
fold_r2_bc_list = []

bias_rows = []

###################### HELPER: PRETRAIN ONCE PER OUTER FOLD #########################

def pretrain_encoder_once_per_fold(train_graphs, fold_id, batch_size, contrastive_epochs, contrastive_lr):
    print(f"\n  [Fold {fold_id}] Contrastive pretraining once on outer-train set")

    seed_everything(1000 + fold_id)

    contrastive_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    encoder = GraphEncoder().to(device)
    contrastive_model = ContrastiveModel(encoder).to(device)

    optimizer_contrastive = torch.optim.AdamW(
        contrastive_model.parameters(),
        lr=contrastive_lr,
        weight_decay=1e-4
    )

    for epoch in range(contrastive_epochs):
        contrastive_loss = train_contrastive_epoch(
            contrastive_model,
            contrastive_loader,
            optimizer_contrastive,
            temperature=0.2
        )

        if (epoch + 1) % 20 == 0:
            print(
                f"      Fold {fold_id} contrastive epoch "
                f"{epoch+1}/{contrastive_epochs} | Loss: {contrastive_loss:.4f}"
            )

    pretrained_encoder_state = copy.deepcopy(encoder.state_dict())
    return pretrained_encoder_state

###################### OUTER CV #########################

for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f"\n================ FOLD {fold+1}/{k} ================")

    outer_train_data = [graph_data_list_addecode[i] for i in train_idx]
    outer_test_data = [graph_data_list_addecode[i] for i in test_idx]

    fold_train_losses = []
    fold_val_losses = []

    repeat_mae_raw = []
    repeat_rmse_raw = []
    repeat_r2_raw = []

    repeat_mae_bc = []
    repeat_rmse_bc = []
    repeat_r2_bc = []

    pretrained_encoder_state = pretrain_encoder_once_per_fold(
        train_graphs=outer_train_data,
        fold_id=fold + 1,
        batch_size=batch_size,
        contrastive_epochs=contrastive_epochs,
        contrastive_lr=contrastive_lr
    )

    for repeat in range(repeats_per_fold):
        print(f"  > Repeat {repeat+1}/{repeats_per_fold}")

        run_seed = 42 + fold * 100 + repeat
        seed_everything(run_seed)

        subtrain_data, val_data = make_inner_train_val_split(
            outer_train_data,
            val_fraction=0.15,
            random_state=run_seed
        )

        subtrain_loader = DataLoader(subtrain_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        outer_train_loader_eval = DataLoader(outer_train_data, batch_size=batch_size, shuffle=False)
        outer_test_loader = DataLoader(outer_test_data, batch_size=batch_size, shuffle=False)

        repeat_encoder = GraphEncoder().to(device)
        repeat_encoder.load_state_dict(pretrained_encoder_state)

        model = BrainAgeRegressor(repeat_encoder).to(device)

        optimizer_supervised = torch.optim.AdamW(
            model.parameters(),
            lr=0.002,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_supervised, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        best_val_loss = float("inf")
        patience_counter = 0

        train_losses = []
        val_losses = []

        best_model_path = os.path.join(
            output_dir,
            f"best_supervised_model_fold_{fold+1}_rep_{repeat+1}.pt"
        )

        for epoch in range(supervised_epochs):
            train_loss = train_supervised_epoch(model, subtrain_loader, optimizer_supervised, criterion)
            val_loss = evaluate_supervised(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"      Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        best_encoder = GraphEncoder()
        best_model = BrainAgeRegressor(best_encoder).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()

        y_train_true, y_train_pred_raw, _ = get_predictions(best_model, outer_train_loader_eval)
        a, b = fit_bias_correction(y_train_true, y_train_pred_raw)

        y_test_true, y_test_pred_raw, ids_test = get_predictions(best_model, outer_test_loader)
        y_test_pred_bc = apply_bias_correction(y_test_pred_raw, a, b)

        all_y_true_raw.extend(y_test_true.tolist())
        all_y_pred_raw.extend(y_test_pred_raw.tolist())
        all_y_pred_bc.extend(y_test_pred_bc.tolist())
        all_subject_ids.extend(ids_test)

        mae_raw = mean_absolute_error(y_test_true, y_test_pred_raw)
        rmse_raw = np.sqrt(mean_squared_error(y_test_true, y_test_pred_raw))
        r2_raw = r2_score(y_test_true, y_test_pred_raw)

        mae_bc = mean_absolute_error(y_test_true, y_test_pred_bc)
        rmse_bc = np.sqrt(mean_squared_error(y_test_true, y_test_pred_bc))
        r2_bc = r2_score(y_test_true, y_test_pred_bc)

        repeat_mae_raw.append(mae_raw)
        repeat_rmse_raw.append(rmse_raw)
        repeat_r2_raw.append(r2_raw)

        repeat_mae_bc.append(mae_bc)
        repeat_rmse_bc.append(rmse_bc)
        repeat_r2_bc.append(r2_bc)

        raw_bag_slope, raw_bag_intercept = compute_bag_slope(y_test_true, y_test_pred_raw)
        bc_bag_slope, bc_bag_intercept = compute_bag_slope(y_test_true, y_test_pred_bc)

        bias_rows.append({
            "Fold": fold + 1,
            "Repeat": repeat + 1,
            "Correction_slope_a": a,
            "Correction_intercept_b": b,
            "Test_BAG_slope_raw": raw_bag_slope,
            "Test_BAG_intercept_raw": raw_bag_intercept,
            "Test_BAG_slope_bias_corrected": bc_bag_slope,
            "Test_BAG_intercept_bias_corrected": bc_bag_intercept
        })

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

    fold_mae_raw_list.append(repeat_mae_raw)
    fold_rmse_raw_list.append(repeat_rmse_raw)
    fold_r2_raw_list.append(repeat_r2_raw)

    fold_mae_bc_list.append(repeat_mae_bc)
    fold_rmse_bc_list.append(repeat_rmse_bc)
    fold_r2_bc_list.append(repeat_r2_bc)

###################### GLOBAL METRICS #########################

all_y_true_raw = np.array(all_y_true_raw)
all_y_pred_raw = np.array(all_y_pred_raw)
all_y_pred_bc = np.array(all_y_pred_bc)

all_maes_raw = np.concatenate(fold_mae_raw_list)
all_rmses_raw = np.concatenate(fold_rmse_raw_list)
all_r2s_raw = np.concatenate(fold_r2_raw_list)

all_maes_bc = np.concatenate(fold_mae_bc_list)
all_rmses_bc = np.concatenate(fold_rmse_bc_list)
all_r2s_bc = np.concatenate(fold_r2_bc_list)

global_raw_bag_slope, global_raw_bag_intercept = compute_bag_slope(all_y_true_raw, all_y_pred_raw)
global_bc_bag_slope, global_bc_bag_intercept = compute_bag_slope(all_y_true_raw, all_y_pred_bc)

print("\n================== FINAL METRICS: RAW ==================")
print(f"Global MAE:  {all_maes_raw.mean():.2f} ± {all_maes_raw.std():.2f}")
print(f"Global RMSE: {all_rmses_raw.mean():.2f} ± {all_rmses_raw.std():.2f}")
print(f"Global R²:   {all_r2s_raw.mean():.2f} ± {all_r2s_raw.std():.2f}")
print(f"Global BAG slope RAW: {global_raw_bag_slope:.6f}")
print("========================================================")

print("\n============= FINAL METRICS: BIAS-CORRECTED =============")
print(f"Global MAE:  {all_maes_bc.mean():.2f} ± {all_maes_bc.std():.2f}")
print(f"Global RMSE: {all_rmses_bc.mean():.2f} ± {all_rmses_bc.std():.2f}")
print(f"Global R²:   {all_r2s_bc.mean():.2f} ± {all_r2s_bc.std():.2f}")
print(f"Global BAG slope BIAS-CORRECTED: {global_bc_bag_slope:.6f}")
print("=========================================================")

###################### SAVE METRICS #########################

metrics_raw_df = pd.DataFrame({
    "MAE_mean": [all_maes_raw.mean()],
    "MAE_std": [all_maes_raw.std()],
    "RMSE_mean": [all_rmses_raw.mean()],
    "RMSE_std": [all_rmses_raw.std()],
    "R2_mean": [all_r2s_raw.mean()],
    "R2_std": [all_r2s_raw.std()],
    "Residual_BAG_slope": [global_raw_bag_slope],
    "Residual_BAG_intercept": [global_raw_bag_intercept]
})

metrics_bc_df = pd.DataFrame({
    "MAE_mean": [all_maes_bc.mean()],
    "MAE_std": [all_maes_bc.std()],
    "RMSE_mean": [all_rmses_bc.mean()],
    "RMSE_std": [all_rmses_bc.std()],
    "R2_mean": [all_r2s_bc.mean()],
    "R2_std": [all_r2s_bc.std()],
    "Residual_BAG_slope": [global_bc_bag_slope],
    "Residual_BAG_intercept": [global_bc_bag_intercept]
})

metrics_raw_path = os.path.join(output_dir, "ADDECODE_SHAPguidedCL_global_metrics_raw.csv")
metrics_bc_path = os.path.join(output_dir, "ADDECODE_SHAPguidedCL_global_metrics_bias_corrected.csv")

metrics_raw_df.to_csv(metrics_raw_path, index=False)
metrics_bc_df.to_csv(metrics_bc_path, index=False)

print("Saved RAW metrics CSV:", metrics_raw_path)
print("Saved BIAS-CORRECTED metrics CSV:", metrics_bc_path)

###################### SAVE BIAS DETAILS #########################

bias_df = pd.DataFrame(bias_rows)
bias_df.to_csv(os.path.join(output_dir, "bias_correction_details_per_fold_repeat.csv"), index=False)

###################### SAVE PREDICTIONS + SUBJECT-LEVEL AGGREGATION #########################

df_preds_all = pd.DataFrame({
    "Subject_ID": all_subject_ids,
    "Real_Age": all_y_true_raw,
    "Predicted_Age_RAW": all_y_pred_raw,
    "Predicted_Age_BiasCorrected": all_y_pred_bc,
})

age_consistency = df_preds_all.groupby("Subject_ID")["Real_Age"].nunique()
if (age_consistency > 1).any():
    raise ValueError("Some subjects have inconsistent Real_Age values across repeats.")

df_subject_level = (
    df_preds_all
    .groupby("Subject_ID", as_index=False)
    .agg({
        "Real_Age": "first",
        "Predicted_Age_RAW": "mean",
        "Predicted_Age_BiasCorrected": "mean"
    })
)

df_subject_level["Brain_Age_Gap_RAW"] = (
    df_subject_level["Predicted_Age_RAW"] - df_subject_level["Real_Age"]
)
df_subject_level["Brain_Age_Gap_BiasCorrected"] = (
    df_subject_level["Predicted_Age_BiasCorrected"] - df_subject_level["Real_Age"]
)

df_preds_all.to_csv(
    os.path.join(output_dir, "cv_predictions_all_repeats.csv"),
    index=False
)

df_subject_level.to_csv(
    os.path.join(output_dir, "cv_predictions_subject_level_aggregated.csv"),
    index=False
)

###################### SUBJECT-LEVEL AGGREGATED CV METRICS #########################

r2_raw_subject = r2_score(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_RAW"]
)
mae_raw_subject = mean_absolute_error(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_RAW"]
)
rmse_raw_subject = np.sqrt(mean_squared_error(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_RAW"]
))

r2_bc_subject = r2_score(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_BiasCorrected"]
)
mae_bc_subject = mean_absolute_error(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_BiasCorrected"]
)
rmse_bc_subject = np.sqrt(mean_squared_error(
    df_subject_level["Real_Age"],
    df_subject_level["Predicted_Age_BiasCorrected"]
))

global_raw_bag_slope_subject, global_raw_bag_intercept_subject = compute_bag_slope(
    df_subject_level["Real_Age"].values,
    df_subject_level["Predicted_Age_RAW"].values
)

global_bc_bag_slope_subject, global_bc_bag_intercept_subject = compute_bag_slope(
    df_subject_level["Real_Age"].values,
    df_subject_level["Predicted_Age_BiasCorrected"].values
)

print("\n========== SUBJECT-LEVEL AGGREGATED CV METRICS: RAW ==========")
print(f"MAE:  {mae_raw_subject:.2f}")
print(f"RMSE: {rmse_raw_subject:.2f}")
print(f"R²:   {r2_raw_subject:.4f}")
print(f"BAG slope: {global_raw_bag_slope_subject:.6f}")
print("==============================================================")

print("\n===== SUBJECT-LEVEL AGGREGATED CV METRICS: BIAS-CORRECTED =====")
print(f"MAE:  {mae_bc_subject:.2f}")
print(f"RMSE: {rmse_bc_subject:.2f}")
print(f"R²:   {r2_bc_subject:.4f}")
print(f"BAG slope: {global_bc_bag_slope_subject:.6f}")
print("==============================================================")

metrics_subject_level_df = pd.DataFrame({
    "Metric_Type": ["RAW", "BiasCorrected"],
    "MAE": [mae_raw_subject, mae_bc_subject],
    "RMSE": [rmse_raw_subject, rmse_bc_subject],
    "R2": [r2_raw_subject, r2_bc_subject],
    "Residual_BAG_slope": [
        global_raw_bag_slope_subject,
        global_bc_bag_slope_subject
    ],
    "Residual_BAG_intercept": [
        global_raw_bag_intercept_subject,
        global_bc_bag_intercept_subject
    ]
})

metrics_subject_level_df.to_csv(
    os.path.join(output_dir, "subject_level_aggregated_metrics.csv"),
    index=False
)

###################### LEARNING CURVES #########################

plt.figure(figsize=(10, 6))
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses[fold][rep], linestyle="dashed", alpha=0.4)
        plt.plot(all_val_losses[fold][rep], alpha=0.4)
plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")
plt.title("Supervised Learning Curves (All Folds / Repeats)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SHAPguidedCL_supervised_learning_curves_all.png"), dpi=300)
plt.close()

###################### SCATTER RAW #########################

plt.figure(figsize=(8, 6))
plt.scatter(df_subject_level["Real_Age"], df_subject_level["Predicted_Age_RAW"], alpha=0.7, edgecolors="k", label="Subject-level RAW")

min_val = min(df_subject_level["Real_Age"].min(), df_subject_level["Predicted_Age_RAW"].min())
max_val = max(df_subject_level["Real_Age"].max(), df_subject_level["Predicted_Age_RAW"].max())
margin = (max_val - min_val) * 0.05

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")

reg_raw = LinearRegression().fit(df_subject_level["Real_Age"].values.reshape(-1, 1), df_subject_level["Predicted_Age_RAW"].values)
x_vals = np.array([min_val, max_val]).reshape(-1, 1)
y_vals = reg_raw.predict(x_vals)
plt.plot(
    x_vals,
    y_vals,
    color="blue",
    alpha=0.5,
    linewidth=2,
    label=f"Trend: y={reg_raw.coef_[0]:.2f}x+{reg_raw.intercept_:.2f}"
)

plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

textstr = (
    f"MAE: {mae_raw_subject:.2f}\n"
    f"RMSE: {rmse_raw_subject:.2f}\n"
    f"R²: {r2_raw_subject:.2f}\n"
    f"BAG slope: {global_raw_bag_slope_subject:.4f}"
)

plt.text(
    0.95, 0.05, textstr,
    transform=plt.gca().transAxes,
    fontsize=11,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
)

plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Ages (RAW, Subject-level)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SHAPguidedCL_scatter_raw_subjectlevel.png"), dpi=300)
plt.close()

###################### SCATTER BIAS-CORRECTED #########################

plt.figure(figsize=(8, 6))
plt.scatter(df_subject_level["Real_Age"], df_subject_level["Predicted_Age_BiasCorrected"], alpha=0.7, edgecolors="k", label="Subject-level Bias-Corrected")

min_val = min(df_subject_level["Real_Age"].min(), df_subject_level["Predicted_Age_BiasCorrected"].min())
max_val = max(df_subject_level["Real_Age"].max(), df_subject_level["Predicted_Age_BiasCorrected"].max())
margin = (max_val - min_val) * 0.05

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")

reg_bc = LinearRegression().fit(df_subject_level["Real_Age"].values.reshape(-1, 1), df_subject_level["Predicted_Age_BiasCorrected"].values)
x_vals = np.array([min_val, max_val]).reshape(-1, 1)
y_vals = reg_bc.predict(x_vals)
plt.plot(
    x_vals,
    y_vals,
    color="blue",
    alpha=0.5,
    linewidth=2,
    label=f"Trend: y={reg_bc.coef_[0]:.2f}x+{reg_bc.intercept_:.2f}"
)

plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

textstr = (
    f"MAE: {mae_bc_subject:.2f}\n"
    f"RMSE: {rmse_bc_subject:.2f}\n"
    f"R²: {r2_bc_subject:.2f}\n"
    f"BAG slope: {global_bc_bag_slope_subject:.4f}"
)

plt.text(
    0.95, 0.05, textstr,
    transform=plt.gca().transAxes,
    fontsize=11,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
)

plt.xlabel("Real Age")
plt.ylabel("Predicted Age (Bias-Corrected)")
plt.title("Predicted vs Real Ages (Bias-Corrected, Subject-level)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SHAPguidedCL_scatter_bias_corrected_subjectlevel.png"), dpi=300)
plt.close()

###################### FINAL MODEL TRAINING ON ALL HEALTHY SUBJECTS #########################

print("\n=== Training final SHAP-guided contrastive model on all healthy subjects ===")

final_encoder = GraphEncoder().to(device)
final_contrastive_model = ContrastiveModel(final_encoder).to(device)

final_contrastive_loader = DataLoader(graph_data_list_addecode, batch_size=batch_size, shuffle=True)
optimizer_contrastive = torch.optim.AdamW(
    final_contrastive_model.parameters(),
    lr=contrastive_lr,
    weight_decay=1e-4
)

for epoch in range(contrastive_epochs):
    loss = train_contrastive_epoch(
        final_contrastive_model,
        final_contrastive_loader,
        optimizer_contrastive,
        temperature=0.2
    )
    if (epoch + 1) % 20 == 0:
        print(f"  Final contrastive epoch {epoch+1}/{contrastive_epochs} | Loss: {loss:.4f}")

final_model = BrainAgeRegressor(final_encoder).to(device)
final_train_loader = DataLoader(graph_data_list_addecode, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = torch.nn.SmoothL1Loss(beta=1)

final_supervised_epochs = 100

for epoch in range(final_supervised_epochs):
    loss = train_supervised_epoch(final_model, final_train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{final_supervised_epochs} | Loss: {loss:.4f}")
    scheduler.step()

final_model_path = os.path.join(output_dir, "final_model_trained_on_all_healthy.pt")
torch.save(final_model.state_dict(), final_model_path)
print("\nFinal model saved as:", final_model_path)

###################### FINAL HEALTHY BIAS MODEL #########################

print("\n=== Fitting final healthy-derived bias correction ===")

final_model.eval()
healthy_loader_for_bias = DataLoader(graph_data_list_addecode, batch_size=batch_size, shuffle=False)
y_healthy_true_final, y_healthy_pred_final, healthy_ids_final = get_predictions(final_model, healthy_loader_for_bias)
a_final, b_final = fit_bias_correction(y_healthy_true_final, y_healthy_pred_final)

print(f"Final healthy bias correction parameters: a={a_final:.6f}, b={b_final:.6f}")

pd.DataFrame({
    "a_final": [a_final],
    "b_final": [b_final]
}).to_csv(os.path.join(output_dir, "final_healthy_bias_correction_parameters.csv"), index=False)

###################### FINAL INFERENCE ON ALL SUBJECTS #########################

print("\n=== Running final model on ALL subjects ===")

all_loader = DataLoader(graph_data_list_all, batch_size=batch_size, shuffle=False)
y_all_true, y_all_pred_raw, ids_all = get_predictions(final_model, all_loader)
y_all_pred_bc = apply_bias_correction(y_all_pred_raw, a_final, b_final)

risk_map_all_graphs = {d.subject_id: getattr(d, "risk", "Unknown") for d in graph_data_list_all}

df_all_predictions = pd.DataFrame({
    "Subject_ID": ids_all,
    "Real_Age": y_all_true,
    "Predicted_Age_RAW": y_all_pred_raw,
    "Predicted_Age_BiasCorrected": y_all_pred_bc,
})

df_all_predictions["Brain_Age_Gap_RAW"] = (
    df_all_predictions["Predicted_Age_RAW"] - df_all_predictions["Real_Age"]
)
df_all_predictions["Brain_Age_Gap_BiasCorrected"] = (
    df_all_predictions["Predicted_Age_BiasCorrected"] - df_all_predictions["Real_Age"]
)
df_all_predictions["Risk"] = df_all_predictions["Subject_ID"].map(risk_map_all_graphs)

df_all_predictions.to_csv(
    os.path.join(output_dir, "final_model_predictions_all_subjects.csv"),
    index=False
)

###################### GROUP SUMMARY FOR ALL SUBJECTS #########################

group_summary = df_all_predictions.groupby("Risk").agg({
    "Real_Age": ["count", "mean", "std"],
    "Predicted_Age_RAW": ["mean", "std"],
    "Predicted_Age_BiasCorrected": ["mean", "std"],
    "Brain_Age_Gap_RAW": ["mean", "std"],
    "Brain_Age_Gap_BiasCorrected": ["mean", "std"],
}).round(3)

group_summary.to_csv(
    os.path.join(output_dir, "final_model_predictions_all_subjects_group_summary.csv")
)

print("\n=== Group summary on ALL subjects ===")
print(group_summary)

###################### OPTIONAL PLOTS FOR ALL SUBJECTS #########################

plt.figure(figsize=(9, 6))
sns.boxplot(data=df_all_predictions, x="Risk", y="Brain_Age_Gap_BiasCorrected")
plt.title("Bias-corrected BAG by Risk group")
plt.ylabel("cBAG")
plt.xlabel("Risk group")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cBAG_boxplot_by_risk_all_subjects.png"), dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
sns.violinplot(data=df_all_predictions, x="Risk", y="Brain_Age_Gap_BiasCorrected", inner="box")
plt.title("Bias-corrected BAG distribution by Risk group")
plt.ylabel("cBAG")
plt.xlabel("Risk group")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cBAG_violin_by_risk_all_subjects.png"), dpi=300)
plt.close()

print("\nSaved all-subject inference outputs.")





###################### SETTINGS #########################

save_raw_plots = False   # set True if you want both RAW and bias-corrected
save_bc_plots = True     # bias-corrected only

###################### PREP DATA FOR PLOTTING #########################

# Original APOE labels from metadata
apoe_label_map = (
    df_matched_all[["MRI_Exam_fixed", "genotype"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "genotype": "APOE_genotype"})
)

df_all_predictions_plot = df_all_predictions.merge(apoe_label_map, on="Subject_ID", how="left")

###################### HELPER FUNCTION #########################

def make_colored_scatter_and_save_metrics(
    df,
    y_col,
    hue_col,
    title,
    out_png,
    metric_label
):
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="Real_Age",
        y=y_col,
        hue=hue_col,
        s=90,
        alpha=0.85,
        edgecolor="black"
    )

    min_val = min(df["Real_Age"].min(), df[y_col].min())
    max_val = max(df["Real_Age"].max(), df[y_col].max())
    margin = (max_val - min_val) * 0.05

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ideal (y=x)"
    )

    reg = LinearRegression().fit(
        df["Real_Age"].values.reshape(-1, 1),
        df[y_col].values
    )
    x_vals = np.array([min_val, max_val]).reshape(-1, 1)
    y_vals = reg.predict(x_vals)

    plt.plot(
        x_vals,
        y_vals,
        color="blue",
        alpha=0.6,
        linewidth=2,
        label=f"Trend: y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}"
    )

    mae_val = mean_absolute_error(df["Real_Age"], df[y_col])
    rmse_val = np.sqrt(mean_squared_error(df["Real_Age"], df[y_col]))
    r2_val = r2_score(df["Real_Age"], df[y_col])
    bag_slope_val, bag_intercept_val = compute_bag_slope(
        df["Real_Age"].values,
        df[y_col].values
    )

    textstr = (
        f"MAE: {mae_val:.2f}\n"
        f"RMSE: {rmse_val:.2f}\n"
        f"R²: {r2_val:.2f}\n"
        f"BAG slope: {bag_slope_val:.4f}"
    )

    plt.text(
        0.97,
        0.03,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
    )

    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    plt.xlabel("Real Age")
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_png), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "Plot": metric_label,
        "Target": y_col,
        "Color_By": hue_col,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "R2": r2_val,
        "BAG_slope": bag_slope_val,
        "BAG_intercept": bag_intercept_val,
        "Trend_slope": float(reg.coef_[0]),
        "Trend_intercept": float(reg.intercept_)
    }

###################### MAKE PLOTS + SAVE METRICS #########################

plot_metrics_rows = []

if save_bc_plots:
    plot_metrics_rows.append(
        make_colored_scatter_and_save_metrics(
            df=df_all_predictions_plot,
            y_col="Predicted_Age_BiasCorrected",
            hue_col="Risk",
            title="Predicted vs Real Age (All Subjects, Bias-Corrected, colored by Diagnosis)",
            out_png="scatter_all_subjects_bias_corrected_colored_by_diagnosis.png",
            metric_label="BC_colored_by_diagnosis"
        )
    )

    plot_metrics_rows.append(
        make_colored_scatter_and_save_metrics(
            df=df_all_predictions_plot,
            y_col="Predicted_Age_BiasCorrected",
            hue_col="APOE_genotype",
            title="Predicted vs Real Age (All Subjects, Bias-Corrected, colored by APOE genotype)",
            out_png="scatter_all_subjects_bias_corrected_colored_by_APOE.png",
            metric_label="BC_colored_by_APOE"
        )
    )

if save_raw_plots:
    plot_metrics_rows.append(
        make_colored_scatter_and_save_metrics(
            df=df_all_predictions_plot,
            y_col="Predicted_Age_RAW",
            hue_col="Risk",
            title="Predicted vs Real Age (All Subjects, RAW, colored by Diagnosis)",
            out_png="scatter_all_subjects_raw_colored_by_diagnosis.png",
            metric_label="RAW_colored_by_diagnosis"
        )
    )

    plot_metrics_rows.append(
        make_colored_scatter_and_save_metrics(
            df=df_all_predictions_plot,
            y_col="Predicted_Age_RAW",
            hue_col="APOE_genotype",
            title="Predicted vs Real Age (All Subjects, RAW, colored by APOE genotype)",
            out_png="scatter_all_subjects_raw_colored_by_APOE.png",
            metric_label="RAW_colored_by_APOE"
        )
    )

plot_metrics_df = pd.DataFrame(plot_metrics_rows)
plot_metrics_df.to_csv(
    os.path.join(output_dir, "all_subjects_scatter_plot_metrics.csv"),
    index=False
)

print("Saved plot metrics to:", os.path.join(output_dir, "all_subjects_scatter_plot_metrics.csv"))





###################### AUC / ROC / CONFUSION MATRIX ANALYSIS #########################

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

###################### MERGE LABELS #########################

# APOE carrier status from explicit APOE column
apoe_status_map = (
    df_matched_all[["MRI_Exam_fixed", "APOE"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "APOE": "APOE_status"})
)

# Full genotype label for plotting
apoe_genotype_map = (
    df_matched_all[["MRI_Exam_fixed", "genotype"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "genotype": "APOE_genotype"})
)

# Sex label
sex_label_map = (
    df_matched_all[["MRI_Exam_fixed", "sex"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "sex": "sex_label"})
)


###################### MERGE LABELS #########################

# APOE carrier status from explicit APOE column
apoe_status_map = (
    df_matched_all[["MRI_Exam_fixed", "APOE"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "APOE": "APOE_status"})
)

# Full genotype label for plotting
apoe_genotype_map = (
    df_matched_all[["MRI_Exam_fixed", "genotype"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "genotype": "APOE_genotype"})
)

# Sex label
sex_label_map = (
    df_matched_all[["MRI_Exam_fixed", "sex"]]
    .drop_duplicates(subset="MRI_Exam_fixed")
    .rename(columns={"MRI_Exam_fixed": "Subject_ID", "sex": "sex_label"})
)

# Build df_auc from final subject-level predictions
df_auc = df_all_predictions.merge(apoe_status_map, on="Subject_ID", how="left")
df_auc = df_auc.merge(apoe_genotype_map, on="Subject_ID", how="left")
df_auc = df_auc.merge(sex_label_map, on="Subject_ID", how="left")

###################### ENCODE LABELS #########################

def encode_apoe4_status(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s == "E4+":
        return 1
    elif s == "E4-":
        return 0
    return np.nan

def encode_sex_binary(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["male", "m", "man"]:
        return 1
    elif s in ["female", "f", "woman"]:
        return 0
    return np.nan

df_auc["APOE4_carrier"] = df_auc["APOE_status"].apply(encode_apoe4_status)
df_auc["Sex_binary"] = df_auc["sex_label"].apply(encode_sex_binary)

###################### RISK GROUPING #########################

# 0 = NoRisk/Familial
# 1 = MCI/AD
risk_binary_map = {
    "NoRisk": 0,
    "Familial": 0,
    "MCI": 1,
    "AD": 1
}
df_auc["Risk_binary_01_vs_23"] = df_auc["Risk"].map(risk_binary_map)


###################### BAG PLOT TABLE #########################

df_bag_plot = df_auc.copy()

# cBAG and mean age for Bland–Altman-style plots
df_bag_plot["cBAG"] = df_bag_plot["Brain_Age_Gap_BiasCorrected"]
df_bag_plot["Mean_Age_BC"] = (
    df_bag_plot["Real_Age"] + df_bag_plot["Predicted_Age_BiasCorrected"]
) / 2.0

# Friendly APOE4 labels
df_bag_plot["APOE4_status_label"] = df_bag_plot["APOE4_carrier"].map({
    0: "E4-",
    1: "E4+"
})

df_bag_plot.to_csv(
    os.path.join(output_dir, "bag_plot_inputs.csv"),
    index=False
)

###################### BOXPLOT: cBAG BY DIAGNOSIS #########################

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df_bag_plot,
    x="Risk",
    y="cBAG"
)
sns.stripplot(
    data=df_bag_plot,
    x="Risk",
    y="cBAG",
    color="black",
    alpha=0.6,
    size=5,
    jitter=True
)
plt.axhline(0, linestyle="--", color="red", linewidth=1.5)
plt.xlabel("Diagnosis")
plt.ylabel("cBAG")
plt.title("Bias-corrected Brain Age Gap by Diagnosis")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "boxplot_cBAG_by_diagnosis.png"),
    dpi=300
)
plt.close()

###################### BOXPLOT: cBAG BY APOE4 CARRIAGE #########################

plt.figure(figsize=(7, 6))
sns.boxplot(
    data=df_bag_plot.dropna(subset=["APOE4_status_label"]),
    x="APOE4_status_label",
    y="cBAG"
)
sns.stripplot(
    data=df_bag_plot.dropna(subset=["APOE4_status_label"]),
    x="APOE4_status_label",
    y="cBAG",
    color="black",
    alpha=0.6,
    size=5,
    jitter=True
)
plt.axhline(0, linestyle="--", color="red", linewidth=1.5)
plt.xlabel("APOE4 carriage")
plt.ylabel("cBAG")
plt.title("Bias-corrected Brain Age Gap by APOE4 Carriage")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "boxplot_cBAG_by_APOE4_carriage.png"),
    dpi=300
)
plt.close()

###################### BLAND-ALTMAN STYLE: cBAG BY DIAGNOSIS #########################

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_bag_plot,
    x="Mean_Age_BC",
    y="cBAG",
    hue="Risk",
    s=90,
    alpha=0.85,
    edgecolor="black"
)

plt.axhline(0, linestyle="--", color="red", linewidth=1.5)

cbag_mean_diag = df_bag_plot["cBAG"].mean()
cbag_std_diag = df_bag_plot["cBAG"].std()
loa_upper_diag = cbag_mean_diag + 1.96 * cbag_std_diag
loa_lower_diag = cbag_mean_diag - 1.96 * cbag_std_diag

plt.axhline(cbag_mean_diag, linestyle="-", linewidth=1.5, color="gray", label=f"Mean cBAG = {cbag_mean_diag:.2f}")
plt.axhline(loa_upper_diag, linestyle=":", linewidth=1.5, color="gray", label=f"+1.96 SD = {loa_upper_diag:.2f}")
plt.axhline(loa_lower_diag, linestyle=":", linewidth=1.5, color="gray", label=f"-1.96 SD = {loa_lower_diag:.2f}")

plt.xlabel("Mean of Real Age and Bias-Corrected Predicted Age")
plt.ylabel("cBAG")
plt.title("Bland–Altman-style Plot of cBAG by Diagnosis")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "bland_altman_cBAG_by_diagnosis.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

###################### BLAND-ALTMAN STYLE: cBAG BY APOE4 CARRIAGE #########################

df_ba_apoe = df_bag_plot.dropna(subset=["APOE4_status_label"]).copy()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_ba_apoe,
    x="Mean_Age_BC",
    y="cBAG",
    hue="APOE4_status_label",
    s=90,
    alpha=0.85,
    edgecolor="black"
)

plt.axhline(0, linestyle="--", color="red", linewidth=1.5)

cbag_mean_apoe = df_ba_apoe["cBAG"].mean()
cbag_std_apoe = df_ba_apoe["cBAG"].std()
loa_upper_apoe = cbag_mean_apoe + 1.96 * cbag_std_apoe
loa_lower_apoe = cbag_mean_apoe - 1.96 * cbag_std_apoe

plt.axhline(cbag_mean_apoe, linestyle="-", linewidth=1.5, color="gray", label=f"Mean cBAG = {cbag_mean_apoe:.2f}")
plt.axhline(loa_upper_apoe, linestyle=":", linewidth=1.5, color="gray", label=f"+1.96 SD = {loa_upper_apoe:.2f}")
plt.axhline(loa_lower_apoe, linestyle=":", linewidth=1.5, color="gray", label=f"-1.96 SD = {loa_lower_apoe:.2f}")

plt.xlabel("Mean of Real Age and Bias-Corrected Predicted Age")
plt.ylabel("cBAG")
plt.title("Bland–Altman-style Plot of cBAG by APOE4 Carriage")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "bland_altman_cBAG_by_APOE4_carriage.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

###################### SAVE GROUP SUMMARIES #########################

diag_summary = df_bag_plot.groupby("Risk").agg(
    N=("cBAG", "count"),
    cBAG_mean=("cBAG", "mean"),
    cBAG_std=("cBAG", "std"),
    Mean_Age_BC_mean=("Mean_Age_BC", "mean"),
    Mean_Age_BC_std=("Mean_Age_BC", "std")
).reset_index()

apoe_summary = df_bag_plot.dropna(subset=["APOE4_status_label"]).groupby("APOE4_status_label").agg(
    N=("cBAG", "count"),
    cBAG_mean=("cBAG", "mean"),
    cBAG_std=("cBAG", "std"),
    Mean_Age_BC_mean=("Mean_Age_BC", "mean"),
    Mean_Age_BC_std=("Mean_Age_BC", "std")
).reset_index()

diag_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_diagnosis.csv"),
    index=False
)

apoe_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_APOE4_carriage.csv"),
    index=False
)

print("Saved BAG boxplots, Bland–Altman-style plots, and summaries.")

###################### SCORE COLUMN #########################

# Main score for discrimination
score_cols = ["Brain_Age_Gap_BiasCorrected"]

###################### SAVE SUBJECT-LEVEL INPUT TABLE #########################

df_auc.to_csv(
    os.path.join(output_dir, "auc_subject_level_inputs.csv"),
    index=False
)

###################### HELPER: AUC #########################

def compute_auc_table(df, y_true_col, score_cols, analysis_name, outcome_name):
    rows = []

    for score_col in score_cols:
        df_local = df.dropna(subset=[y_true_col, score_col]).copy()

        if df_local.empty or df_local[y_true_col].nunique() < 2:
            rows.append({
                "Analysis": analysis_name,
                "Outcome": outcome_name,
                "Score": score_col,
                "AUC": np.nan,
                "N": len(df_local),
                "N_negative": np.nan,
                "N_positive": np.nan
            })
            continue

        y_true = df_local[y_true_col].astype(int).values
        scores = df_local[score_col].values

        auc_val = roc_auc_score(y_true, scores)

        rows.append({
            "Analysis": analysis_name,
            "Outcome": outcome_name,
            "Score": score_col,
            "AUC": auc_val,
            "N": len(df_local),
            "N_negative": int((df_local[y_true_col] == 0).sum()),
            "N_positive": int((df_local[y_true_col] == 1).sum())
        })

    return pd.DataFrame(rows)

###################### HELPER: ROC PLOT #########################

def make_roc_plot(df, y_true_col, score_cols, title, out_png, positive_label_name):
    plt.figure(figsize=(8, 6))
    roc_rows = []
    plotted_any = False

    for score_col in score_cols:
        df_local = df.dropna(subset=[y_true_col, score_col]).copy()

        if df_local.empty or df_local[y_true_col].nunique() < 2:
            continue

        y_true = df_local[y_true_col].astype(int).values
        scores = df_local[score_col].values

        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{score_col} (AUC = {auc_val:.3f})"
        )

        roc_rows.append({
            "Outcome": positive_label_name,
            "Score": score_col,
            "AUC": auc_val,
            "N": len(df_local),
            "N_negative": int((df_local[y_true_col] == 0).sum()),
            "N_positive": int((df_local[y_true_col] == 1).sum())
        })

        plotted_any = True

    if not plotted_any:
        plt.close()
        print(f"Skipping ROC plot for {title}: not enough class variation.")
        return pd.DataFrame()

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_png), dpi=300)
    plt.close()

    return pd.DataFrame(roc_rows)

###################### HELPER: CONFUSION MATRIX + METRICS #########################

def compute_confusion_metrics_at_best_threshold(df, y_true_col, score_col, analysis_name, outcome_name):
    df_local = df.dropna(subset=[y_true_col, score_col]).copy()

    if df_local.empty or df_local[y_true_col].nunique() < 2:
        return None, None, None

    y_true = df_local[y_true_col].astype(int).values
    scores = df_local[score_col].values

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    y_pred = (scores >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics_row = {
        "Analysis": analysis_name,
        "Outcome": outcome_name,
        "Score": score_col,
        "Threshold": best_threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": accuracy,
        "N": len(df_local)
    }

    cm_df = pd.DataFrame(
        [[tn, fp],
         [fn, tp]],
        index=["True_0", "True_1"],
        columns=["Pred_0", "Pred_1"]
    )

    df_preds = df_local.copy()
    df_preds["Predicted_binary"] = y_pred
    df_preds["Classification"] = np.where(
        (df_preds[y_true_col] == 1) & (df_preds["Predicted_binary"] == 1), "TP",
        np.where(
            (df_preds[y_true_col] == 0) & (df_preds["Predicted_binary"] == 0), "TN",
            np.where(
                (df_preds[y_true_col] == 0) & (df_preds["Predicted_binary"] == 1), "FP",
                "FN"
            )
        )
    )

    return metrics_row, cm_df, df_preds

###################### 1) AUC TABLES #########################

auc_apoe_df = compute_auc_table(
    df=df_auc,
    y_true_col="APOE4_carrier",
    score_cols=score_cols,
    analysis_name="APOE4_carriage",
    outcome_name="E4-_vs_E4+"
)

auc_sex_df = compute_auc_table(
    df=df_auc,
    y_true_col="Sex_binary",
    score_cols=score_cols,
    analysis_name="Sex",
    outcome_name="Female_vs_Male"
)

auc_risk_df = compute_auc_table(
    df=df_auc,
    y_true_col="Risk_binary_01_vs_23",
    score_cols=score_cols,
    analysis_name="Risk_grouping",
    outcome_name="NoRisk_Familial_vs_MCI_AD"
)

auc_results_df = pd.concat([auc_apoe_df, auc_sex_df, auc_risk_df], ignore_index=True)
auc_results_df.to_csv(
    os.path.join(output_dir, "auc_results_apoe4_sex_riskgroup.csv"),
    index=False
)

print("\n=== AUC RESULTS ===")
print(auc_results_df)

###################### 2) ROC CURVES #########################

roc_apoe_df = make_roc_plot(
    df=df_auc,
    y_true_col="APOE4_carrier",
    score_cols=score_cols,
    title="ROC Curve: APOE4 carriage",
    out_png="roc_curve_APOE4_carriage.png",
    positive_label_name="APOE4_carrier"
)

roc_sex_df = make_roc_plot(
    df=df_auc,
    y_true_col="Sex_binary",
    score_cols=score_cols,
    title="ROC Curve: Sex",
    out_png="roc_curve_sex.png",
    positive_label_name="male_positive"
)

roc_risk_df = make_roc_plot(
    df=df_auc,
    y_true_col="Risk_binary_01_vs_23",
    score_cols=score_cols,
    title="ROC Curve: NoRisk/Familial vs MCI/AD",
    out_png="roc_curve_riskgroup_01_vs_23.png",
    positive_label_name="MCI_AD_positive"
)

roc_metrics_df = pd.concat([roc_apoe_df, roc_sex_df, roc_risk_df], ignore_index=True)
roc_metrics_df.to_csv(
    os.path.join(output_dir, "roc_curve_metrics.csv"),
    index=False
)

###################### 3) CONFUSION MATRICES #########################

confusion_metric_rows = []

# APOE4
cm_metrics_apoe, cm_apoe_df, df_apoe_preds = compute_confusion_metrics_at_best_threshold(
    df=df_auc,
    y_true_col="APOE4_carrier",
    score_col="Brain_Age_Gap_BiasCorrected",
    analysis_name="APOE4_carriage",
    outcome_name="E4-_vs_E4+"
)
if cm_metrics_apoe is not None:
    confusion_metric_rows.append(cm_metrics_apoe)
    cm_apoe_df.to_csv(os.path.join(output_dir, "confusion_matrix_APOE4_cBAG.csv"))
    df_apoe_preds.to_csv(os.path.join(output_dir, "classification_table_APOE4_cBAG.csv"), index=False)

# Sex
cm_metrics_sex, cm_sex_df, df_sex_preds = compute_confusion_metrics_at_best_threshold(
    df=df_auc,
    y_true_col="Sex_binary",
    score_col="Brain_Age_Gap_BiasCorrected",
    analysis_name="Sex",
    outcome_name="Female_vs_Male"
)
if cm_metrics_sex is not None:
    confusion_metric_rows.append(cm_metrics_sex)
    cm_sex_df.to_csv(os.path.join(output_dir, "confusion_matrix_sex_cBAG.csv"))
    df_sex_preds.to_csv(os.path.join(output_dir, "classification_table_sex_cBAG.csv"), index=False)

# Risk grouping
cm_metrics_risk, cm_risk_df, df_risk_preds = compute_confusion_metrics_at_best_threshold(
    df=df_auc,
    y_true_col="Risk_binary_01_vs_23",
    score_col="Brain_Age_Gap_BiasCorrected",
    analysis_name="Risk_grouping",
    outcome_name="NoRisk_Familial_vs_MCI_AD"
)
if cm_metrics_risk is not None:
    confusion_metric_rows.append(cm_metrics_risk)
    cm_risk_df.to_csv(os.path.join(output_dir, "confusion_matrix_riskgroup_cBAG.csv"))
    df_risk_preds.to_csv(os.path.join(output_dir, "classification_table_riskgroup_cBAG.csv"), index=False)

confusion_metrics_df = pd.DataFrame(confusion_metric_rows)
confusion_metrics_df.to_csv(
    os.path.join(output_dir, "confusion_metrics_summary.csv"),
    index=False
)

print("\n=== CONFUSION METRICS ===")
print(confusion_metrics_df)

###################### 4) OPTIONAL: COLORED SCATTER PLOTS #########################

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_auc,
    x="Real_Age",
    y="Predicted_Age_BiasCorrected",
    hue="Risk",
    s=90,
    alpha=0.85,
    edgecolor="black"
)
min_val = min(df_auc["Real_Age"].min(), df_auc["Predicted_Age_BiasCorrected"].min())
max_val = max(df_auc["Real_Age"].max(), df_auc["Predicted_Age_BiasCorrected"].max())
margin = (max_val - min_val) * 0.05
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y=x)")
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (Bias-Corrected)")
plt.title("Predicted vs Real Age (All Subjects, colored by Diagnosis)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_all_subjects_bias_corrected_colored_by_diagnosis.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_auc,
    x="Real_Age",
    y="Predicted_Age_BiasCorrected",
    hue="APOE_genotype",
    s=90,
    alpha=0.85,
    edgecolor="black"
)
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y=x)")
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (Bias-Corrected)")
plt.title("Predicted vs Real Age (All Subjects, colored by APOE genotype)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_all_subjects_bias_corrected_colored_by_APOE.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_auc,
    x="Real_Age",
    y="Predicted_Age_BiasCorrected",
    hue="sex_label",
    s=90,
    alpha=0.85,
    edgecolor="black"
)
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y=x)")
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (Bias-Corrected)")
plt.title("Predicted vs Real Age (All Subjects, colored by Sex)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_all_subjects_bias_corrected_colored_by_sex.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved AUC / ROC / confusion matrix outputs.")


###################### cBAG GROUP PLOTS + CSV STATS + EFFECT SIZES #########################

# Requires:
# - df_auc already created
# - output_dir already defined
# - columns available in df_auc:
#   "Brain_Age_Gap_BiasCorrected", "Risk", "APOE_genotype", "sex_label", "Real_Age"

from scipy.stats import mannwhitneyu, kruskal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

###################### PREP DATA #########################

df_cbag = df_auc.copy()
df_cbag["cBAG"] = df_cbag["Brain_Age_Gap_BiasCorrected"]

###################### HELPERS #########################

def cliffs_delta(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return (gt - lt) / (len(x) * len(y))

def rank_biserial_from_mwu(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    res = mannwhitneyu(x, y, alternative="two-sided")
    u = res.statistic
    n1 = len(x)
    n2 = len(y)
    rbc = (2 * u) / (n1 * n2) - 1
    return rbc, res.pvalue

def eta_squared_kruskal(groups):
    clean_groups = [np.asarray(pd.Series(g).dropna(), dtype=float) for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]

    if len(clean_groups) < 2:
        return np.nan, np.nan

    h_stat, pval = kruskal(*clean_groups)
    n_total = sum(len(g) for g in clean_groups)
    k = len(clean_groups)

    if n_total <= k:
        return np.nan, pval

    eta_sq = (h_stat - k + 1) / (n_total - k)
    eta_sq = max(0.0, eta_sq)
    return eta_sq, pval

def add_n_to_categories(df, group_col, order=None):
    counts = df[group_col].value_counts(dropna=False).to_dict()

    if order is None:
        categories = [c for c in df[group_col].dropna().unique()]
    else:
        categories = order

    label_map = {}
    for cat in categories:
        n = counts.get(cat, 0)
        label_map[cat] = f"{cat}\n(n={n})"

    df = df.copy()
    df[f"{group_col}_labelN"] = df[group_col].map(label_map)
    return df, label_map

def summarize_group(df, group_col):
    return (
        df.groupby(group_col)
        .agg(
            N=("cBAG", "count"),
            cBAG_mean=("cBAG", "mean"),
            cBAG_std=("cBAG", "std"),
            cBAG_median=("cBAG", "median"),
            cBAG_q25=("cBAG", lambda x: x.quantile(0.25)),
            cBAG_q75=("cBAG", lambda x: x.quantile(0.75)),
            cBAG_min=("cBAG", "min"),
            cBAG_max=("cBAG", "max"),
        )
        .reset_index()
    )

def make_boxplot(df, x_col, x_label, title, filename, stats_text=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x_col, y="cBAG")
    sns.stripplot(
        data=df,
        x=x_col,
        y="cBAG",
        color="black",
        alpha=0.6,
        size=5,
        jitter=True
    )
    plt.axhline(0, linestyle="--", color="red", linewidth=1.5)
    plt.xlabel(x_label)
    plt.ylabel("cBAG")
    plt.title(title)

    if stats_text is not None:
        plt.text(
            0.98,
            0.02,
            stats_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.9
            )
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

###################### SAVE INPUT TABLE #########################

df_cbag.to_csv(
    os.path.join(output_dir, "cbag_group_plot_input_table.csv"),
    index=False
)

###################### 1) RISK #########################

risk_order = ["NoRisk", "Familial", "MCI", "AD"]
df_risk = df_cbag[df_cbag["Risk"].isin(risk_order)].copy()
df_risk["Risk"] = pd.Categorical(df_risk["Risk"], categories=risk_order, ordered=True)

df_risk, risk_label_map = add_n_to_categories(df_risk, "Risk", order=risk_order)
risk_label_order = [risk_label_map[g] for g in risk_order]

df_risk["Risk_labelN"] = pd.Categorical(
    df_risk["Risk_labelN"],
    categories=risk_label_order,
    ordered=True
)

risk_summary = summarize_group(df_risk, "Risk")
risk_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_risk_withN.csv"),
    index=False
)

risk_groups = [
    df_risk.loc[df_risk["Risk"] == g, "cBAG"].values
    for g in risk_order
    if (df_risk["Risk"] == g).sum() > 0
]
risk_eta_sq, risk_p = eta_squared_kruskal(risk_groups)

risk_effects = pd.DataFrame([{
    "Grouping": "Risk",
    "Test": "Kruskal-Wallis",
    "Effect_size": "Eta_squared_H",
    "Effect_value": risk_eta_sq,
    "p_value": risk_p,
    "Levels": ", ".join([g for g in risk_order if (df_risk["Risk"] == g).sum() > 0])
}])

risk_effects.to_csv(
    os.path.join(output_dir, "effect_size_cBAG_by_risk.csv"),
    index=False
)

risk_stats_text = (
    f"Kruskal–Wallis p = {risk_p:.3g}\n"
    f"Eta²(H) = {risk_eta_sq:.3f}"
)

make_boxplot(
    df=df_risk,
    x_col="Risk_labelN",
    x_label="Diagnosis",
    title="Bias-corrected Brain Age Gap by Diagnosis",
    filename="boxplot_cBAG_by_risk_withN.png",
    stats_text=risk_stats_text
)

###################### 2) APOE GENOTYPE #########################

apoe_order = ["APOE23", "APOE33", "APOE34", "APOE44"]
df_apoe = df_cbag[df_cbag["APOE_genotype"].isin(apoe_order)].copy()
df_apoe["APOE_genotype"] = pd.Categorical(
    df_apoe["APOE_genotype"],
    categories=apoe_order,
    ordered=True
)

df_apoe, apoe_label_map = add_n_to_categories(df_apoe, "APOE_genotype", order=apoe_order)
apoe_label_order = [apoe_label_map[g] for g in apoe_order]

df_apoe["APOE_genotype_labelN"] = pd.Categorical(
    df_apoe["APOE_genotype_labelN"],
    categories=apoe_label_order,
    ordered=True
)

apoe_summary = summarize_group(df_apoe, "APOE_genotype")
apoe_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_APOE_genotype_withN.csv"),
    index=False
)

apoe_groups = [
    df_apoe.loc[df_apoe["APOE_genotype"] == g, "cBAG"].values
    for g in apoe_order
    if (df_apoe["APOE_genotype"] == g).sum() > 0
]
apoe_eta_sq, apoe_p = eta_squared_kruskal(apoe_groups)

apoe_effects = pd.DataFrame([{
    "Grouping": "APOE_genotype",
    "Test": "Kruskal-Wallis",
    "Effect_size": "Eta_squared_H",
    "Effect_value": apoe_eta_sq,
    "p_value": apoe_p,
    "Levels": ", ".join([g for g in apoe_order if (df_apoe["APOE_genotype"] == g).sum() > 0])
}])

apoe_effects.to_csv(
    os.path.join(output_dir, "effect_size_cBAG_by_APOE_genotype.csv"),
    index=False
)

apoe_stats_text = (
    f"Kruskal–Wallis p = {apoe_p:.3g}\n"
    f"Eta²(H) = {apoe_eta_sq:.3f}"
)

make_boxplot(
    df=df_apoe,
    x_col="APOE_genotype_labelN",
    x_label="APOE genotype",
    title="Bias-corrected Brain Age Gap by APOE Genotype",
    filename="boxplot_cBAG_by_APOE_genotype_withN.png",
    stats_text=apoe_stats_text
)

###################### 3) SEX #########################

sex_order = ["F", "M"]
df_sex = df_cbag[df_cbag["sex_label"].isin(sex_order)].copy()
df_sex["sex_label"] = pd.Categorical(df_sex["sex_label"], categories=sex_order, ordered=True)

df_sex, sex_label_map = add_n_to_categories(df_sex, "sex_label", order=sex_order)
sex_label_order = [sex_label_map[g] for g in sex_order]

df_sex["sex_label_labelN"] = pd.Categorical(
    df_sex["sex_label_labelN"],
    categories=sex_label_order,
    ordered=True
)

sex_summary = summarize_group(df_sex, "sex_label")
sex_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_sex_withN.csv"),
    index=False
)

sex_f = df_sex.loc[df_sex["sex_label"] == "F", "cBAG"].values
sex_m = df_sex.loc[df_sex["sex_label"] == "M", "cBAG"].values
sex_rbc, sex_p = rank_biserial_from_mwu(sex_f, sex_m)
sex_cliff = cliffs_delta(sex_f, sex_m)

sex_effects = pd.DataFrame([{
    "Grouping": "Sex",
    "Comparison": "F vs M",
    "Test": "Mann-Whitney U",
    "Effect_size_1": "Rank_biserial_correlation",
    "Effect_value_1": sex_rbc,
    "Effect_size_2": "Cliffs_delta",
    "Effect_value_2": sex_cliff,
    "p_value": sex_p,
    "N_F": len(sex_f),
    "N_M": len(sex_m)
}])

sex_effects.to_csv(
    os.path.join(output_dir, "effect_size_cBAG_by_sex.csv"),
    index=False
)

sex_stats_text = (
    f"Mann–Whitney p = {sex_p:.3g}\n"
    f"Cliff's delta = {sex_cliff:.3f}\n"
    f"Rank-biserial r = {sex_rbc:.3f}"
)

make_boxplot(
    df=df_sex,
    x_col="sex_label_labelN",
    x_label="Sex",
    title="Bias-corrected Brain Age Gap by Sex",
    filename="boxplot_cBAG_by_sex_withN.png",
    stats_text=sex_stats_text
)

###################### 4) MEDIAN AGE SPLIT #########################

df_age = df_cbag.copy()
median_age = df_age["Real_Age"].median()

low_label = f"< median ({median_age:.1f})"
high_label = f"≥ median ({median_age:.1f})"

df_age["Age_group_median"] = np.where(
    df_age["Real_Age"] < median_age,
    low_label,
    high_label
)

age_order = [low_label, high_label]
df_age["Age_group_median"] = pd.Categorical(
    df_age["Age_group_median"],
    categories=age_order,
    ordered=True
)

df_age, age_label_map = add_n_to_categories(df_age, "Age_group_median", order=age_order)
age_label_order = [age_label_map[g] for g in age_order]

df_age["Age_group_median_labelN"] = pd.Categorical(
    df_age["Age_group_median_labelN"],
    categories=age_label_order,
    ordered=True
)

age_summary = summarize_group(df_age, "Age_group_median")
age_summary.to_csv(
    os.path.join(output_dir, "summary_cBAG_by_median_age_withN.csv"),
    index=False
)

age_low = df_age.loc[df_age["Age_group_median"] == low_label, "cBAG"].values
age_high = df_age.loc[df_age["Age_group_median"] == high_label, "cBAG"].values
age_rbc, age_p = rank_biserial_from_mwu(age_low, age_high)
age_cliff = cliffs_delta(age_low, age_high)

age_effects = pd.DataFrame([{
    "Grouping": "Median_age_split",
    "Comparison": f"{low_label} vs {high_label}",
    "Test": "Mann-Whitney U",
    "Effect_size_1": "Rank_biserial_correlation",
    "Effect_value_1": age_rbc,
    "Effect_size_2": "Cliffs_delta",
    "Effect_value_2": age_cliff,
    "p_value": age_p,
    "N_low": len(age_low),
    "N_high": len(age_high),
    "Median_age_cutoff": median_age
}])

age_effects.to_csv(
    os.path.join(output_dir, "effect_size_cBAG_by_median_age.csv"),
    index=False
)

age_stats_text = (
    f"Mann–Whitney p = {age_p:.3g}\n"
    f"Cliff's delta = {age_cliff:.3f}\n"
    f"Rank-biserial r = {age_rbc:.3f}\n"
    f"Median = {median_age:.1f}"
)

make_boxplot(
    df=df_age,
    x_col="Age_group_median_labelN",
    x_label="Age group",
    title="Bias-corrected Brain Age Gap by Median Age Split",
    filename="boxplot_cBAG_by_median_age_withN.png",
    stats_text=age_stats_text
)

###################### COMBINED EFFECT SIZE TABLE #########################

all_effects = pd.concat(
    [risk_effects, apoe_effects, sex_effects, age_effects],
    ignore_index=True,
    sort=False
)

all_effects.to_csv(
    os.path.join(output_dir, "effect_sizes_cBAG_all_groupings.csv"),
    index=False
)

###################### COMBINED SUMMARY TABLE #########################

risk_summary["Grouping"] = "Risk"
risk_summary = risk_summary.rename(columns={"Risk": "Level"})

apoe_summary["Grouping"] = "APOE_genotype"
apoe_summary = apoe_summary.rename(columns={"APOE_genotype": "Level"})

sex_summary["Grouping"] = "Sex"
sex_summary = sex_summary.rename(columns={"sex_label": "Level"})

age_summary["Grouping"] = "Median_age_split"
age_summary = age_summary.rename(columns={"Age_group_median": "Level"})

all_summaries = pd.concat(
    [risk_summary, apoe_summary, sex_summary, age_summary],
    ignore_index=True
)

all_summaries.to_csv(
    os.path.join(output_dir, "summary_cBAG_all_groupings.csv"),
    index=False
)

print("\nSaved:")
print("- cbag_group_plot_input_table.csv")
print("- boxplot_cBAG_by_risk_withN.png")
print("- boxplot_cBAG_by_APOE_genotype_withN.png")
print("- boxplot_cBAG_by_sex_withN.png")
print("- boxplot_cBAG_by_median_age_withN.png")
print("- summary_cBAG_by_risk_withN.csv")
print("- summary_cBAG_by_APOE_genotype_withN.csv")
print("- summary_cBAG_by_sex_withN.csv")
print("- summary_cBAG_by_median_age_withN.csv")
print("- effect_size_cBAG_by_risk.csv")
print("- effect_size_cBAG_by_APOE_genotype.csv")
print("- effect_size_cBAG_by_sex.csv")
print("- effect_size_cBAG_by_median_age.csv")
print("- effect_sizes_cBAG_all_groupings.csv")
print("- summary_cBAG_all_groupings.csv")