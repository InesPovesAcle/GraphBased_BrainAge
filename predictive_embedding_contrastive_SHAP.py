#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:22:13 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:01:29 2026

@author: ines
"""

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

CORRECTIONS APPLIED
- Fixed genotype/global_feature_names inconsistency
- Added robust SHAP plotting helper to avoid blank saved plots
- Replaced fragile embedding SHAP on GraphEmb_0 with embedding SHAP across ALL graph embedding dimensions
- Added debug prints for embedding variance / missing values
- Replaced SHAP summary plot for embedding with stable manual barplot
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
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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
GRAPH_EMB_DIM = 64
# Fit encoders on healthy only, apply to all
le_sex = LabelEncoder()
le_sex.fit(addecode_healthy_metadata_pca["sex"].astype(str))
addecode_healthy_metadata_pca["sex_encoded"] = le_sex.transform(addecode_healthy_metadata_pca["sex"].astype(str))
addecode_all_metadata_pca["sex_encoded"] = le_sex.transform(addecode_all_metadata_pca["sex"].astype(str))

le_genotype = LabelEncoder()
le_genotype.fit(addecode_healthy_metadata_pca["genotype"].astype(str))
addecode_healthy_metadata_pca["genotype_encoded"] = le_genotype.transform(addecode_healthy_metadata_pca["genotype"].astype(str))
addecode_all_metadata_pca["genotype_encoded"] = le_genotype.transform(addecode_all_metadata_pca["genotype"].astype(str))

numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ["PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"]

global_feature_names = [
    "Systolic",
    "Diastolic",
    "sex_encoded",
    "genotype_encoded",
    "Clustering_Coeff",
    "Path_Length"
] + pca_cols

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
        row["genotype_encoded"]
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
        row["genotype_encoded"]
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
    def __init__(self, in_channels=4, hidden_channels=64, graph_emb_dim=GRAPH_EMB_DIM):
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

        self.post_pool = nn.Linear(128, graph_emb_dim)
        

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
    def __init__(self, encoder, graph_emb_dim=64, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(graph_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, proj_dim)
        )

    def forward(self, data):
        h = self.encoder(data)
        z = self.projector(h)
        return h, z

class BrainAgeRegressor(nn.Module):
    def __init__(self, encoder, graph_emb_dim=64):
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
            nn.Linear(graph_emb_dim + 16 + 16 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data, return_parts=False):
        x_graph = self.encoder(data)

        global_feats = data.global_features.to(x_graph.device).squeeze(1)
        meta_embed = self.meta_head(global_feats[:, 0:4])
        graph_embed = self.graph_head(global_feats[:, 4:6])
        pca_embed = self.pca_head(global_feats[:, 6:])

        fused_embedding = torch.cat(
            [x_graph, meta_embed, graph_embed, pca_embed],
            dim=1
        )
        pred = self.fc(fused_embedding)

        if return_parts:
            return {
                "prediction": pred,
                "graph_embedding": x_graph,
                "meta_embedding": meta_embed,
                "graphmetric_embedding": graph_embed,
                "pca_embedding": pca_embed,
                "fused_embedding": fused_embedding,
                "global_features": global_feats
            }

        return pred

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

def get_predictions_and_embeddings(model, loader):
    model.eval()

    rows = []
    fused_list = []
    graph_list = []
    meta_list = []
    graphmetric_list = []
    pca_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, return_parts=True)

            pred = out["prediction"].view(-1).cpu().numpy()
            true = data.y.cpu().numpy()
            ids = [str(sid) for sid in data.subject_id]

            fused = out["fused_embedding"].cpu().numpy()
            graph_emb = out["graph_embedding"].cpu().numpy()
            meta_emb = out["meta_embedding"].cpu().numpy()
            graphmetric_emb = out["graphmetric_embedding"].cpu().numpy()
            pca_emb = out["pca_embedding"].cpu().numpy()

            for i in range(len(ids)):
                rows.append({
                    "Subject_ID": ids[i],
                    "Real_Age": float(true[i]),
                    "Predicted_Age": float(pred[i])
                })

            fused_list.append(fused)
            graph_list.append(graph_emb)
            meta_list.append(meta_emb)
            graphmetric_list.append(graphmetric_emb)
            pca_list.append(pca_emb)

    df = pd.DataFrame(rows)

    fused_mat = np.vstack(fused_list)
    graph_mat = np.vstack(graph_list)
    meta_mat = np.vstack(meta_list)
    graphmetric_mat = np.vstack(graphmetric_list)
    pca_mat = np.vstack(pca_list)

    fused_cols = [f"Fused_{i}" for i in range(fused_mat.shape[1])]
    graph_cols = [f"GraphEmb_{i}" for i in range(graph_mat.shape[1])]
    meta_cols = [f"MetaEmb_{i}" for i in range(meta_mat.shape[1])]
    graphmetric_cols = [f"GraphMetricEmb_{i}" for i in range(graphmetric_mat.shape[1])]
    pca_cols_local = [f"PCAEmb_{i}" for i in range(pca_mat.shape[1])]

    df_fused = pd.DataFrame(fused_mat, columns=fused_cols)
    df_graph = pd.DataFrame(graph_mat, columns=graph_cols)
    df_meta = pd.DataFrame(meta_mat, columns=meta_cols)
    df_graphmetric = pd.DataFrame(graphmetric_mat, columns=graphmetric_cols)
    df_pca = pd.DataFrame(pca_mat, columns=pca_cols_local)

    df_all = pd.concat([df.reset_index(drop=True),
                        df_fused, df_graph, df_meta, df_graphmetric, df_pca], axis=1)

    return df_all

###################### SHAP PLOT HELPERS #########################

def save_shap_bar_plot(shap_values, X, out_path, title=None):
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="bar"
    )
    fig = plt.gcf()
    if title is not None:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_importance_barplot(df_importance, out_path, title, top_n=15):
    df_plot = df_importance.head(top_n).copy()
    plt.figure(figsize=(8, max(5, 0.35 * len(df_plot))))
    sns.barplot(data=df_plot, x="MeanAbsSHAP", y="Feature")
    plt.title(title)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

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
    
    encoder = GraphEncoder(graph_emb_dim=GRAPH_EMB_DIM).to(device)
    contrastive_model = ContrastiveModel(encoder, graph_emb_dim=GRAPH_EMB_DIM).to(device)
    
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
        
       
        repeat_encoder = GraphEncoder(graph_emb_dim=GRAPH_EMB_DIM).to(device)
       
        model = BrainAgeRegressor(repeat_encoder, graph_emb_dim=GRAPH_EMB_DIM).to(device)
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

        best_encoder = GraphEncoder(graph_emb_dim=GRAPH_EMB_DIM)
        best_model = BrainAgeRegressor(best_encoder, graph_emb_dim=GRAPH_EMB_DIM).to(device)
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

final_encoder = GraphEncoder(graph_emb_dim=GRAPH_EMB_DIM).to(device)
final_contrastive_model = ContrastiveModel(final_encoder, graph_emb_dim=GRAPH_EMB_DIM).to(device)

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

final_model = BrainAgeRegressor(final_encoder, graph_emb_dim=GRAPH_EMB_DIM).to(device)
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

###################### SAVE FINAL EMBEDDINGS #########################

final_embeddings_df = get_predictions_and_embeddings(final_model, all_loader)

risk_map_all_graphs = {d.subject_id: getattr(d, "risk", "Unknown") for d in graph_data_list_all}
final_embeddings_df["Risk"] = final_embeddings_df["Subject_ID"].map(risk_map_all_graphs)

final_embeddings_df["Brain_Age_Gap"] = (
    final_embeddings_df["Predicted_Age"] - final_embeddings_df["Real_Age"]
)

final_embeddings_df.to_csv(
    os.path.join(output_dir, "final_model_embeddings_all_subjects.csv"),
    index=False
)

print("Saved final embeddings table.")

###################### PREDICTIVE SHAP #########################

predictive_feature_cols = [c for c in final_embeddings_df.columns if c.startswith("Fused_")]

X_pred = final_embeddings_df[predictive_feature_cols].copy()
y_pred_target = final_embeddings_df["Predicted_Age"].values

rf_pred = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_pred.fit(X_pred, y_pred_target)

explainer_pred = shap.TreeExplainer(rf_pred)
shap_values_pred = explainer_pred.shap_values(X_pred)

predictive_shap_importance = pd.DataFrame({
    "Feature": predictive_feature_cols,
    "MeanAbsSHAP": np.abs(shap_values_pred).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

predictive_shap_importance.to_csv(
    os.path.join(output_dir, "predictive_shap_importance_fused_embedding.csv"),
    index=False
)

save_shap_bar_plot(
    shap_values_pred,
    X_pred,
    os.path.join(output_dir, "Figure_4_predictive_SHAP_bar.png"),
    title="Predictive SHAP importance (fused embedding)"
)

print("Saved predictive SHAP outputs.")

###################### EMBEDDING SHAP #########################

original_feature_df = addecode_all_metadata_pca.copy()

original_feature_df = original_feature_df[[
    "MRI_Exam_fixed",
    "Systolic",
    "Diastolic",
    "sex_encoded",
    "genotype_encoded",
    "Clustering_Coeff",
    "Path_Length"
] + pca_cols].copy()

original_feature_df = original_feature_df.rename(columns={
    "MRI_Exam_fixed": "Subject_ID"
})

original_feature_df = original_feature_df.drop_duplicates(subset="Subject_ID")

embedding_merge_df = final_embeddings_df.merge(
    original_feature_df,
    on="Subject_ID",
    how="left"
)

embedding_input_cols = [
    "Systolic",
    "Diastolic",
    "sex_encoded",
    "genotype_encoded",
    "Clustering_Coeff",
    "Path_Length"
] + pca_cols

graph_emb_cols = [c for c in final_embeddings_df.columns if c.startswith("GraphEmb_")]

X_emb = embedding_merge_df[embedding_input_cols].copy()
Y_emb = embedding_merge_df[graph_emb_cols].copy()

print("\n=== EMBEDDING SHAP DEBUG ===")
print("X_emb shape:", X_emb.shape)
print("Y_emb shape:", Y_emb.shape)
print("NaNs in X_emb:")
print(X_emb.isna().sum())
print("Lowest embedding variances:")
print(Y_emb.var().sort_values().head(10))
print("Highest embedding variances:")
print(Y_emb.var().sort_values(ascending=False).head(10))

valid_mask = ~(X_emb.isna().any(axis=1) | Y_emb.isna().any(axis=1))
X_emb = X_emb.loc[valid_mask].reset_index(drop=True)
Y_emb = Y_emb.loc[valid_mask].reset_index(drop=True)

all_dim_importances = []
per_dim_rows = []

for target_col in Y_emb.columns:
    y_emb = Y_emb[target_col].values

    if np.nanstd(y_emb) < 1e-8:
        print(f"Skipping {target_col}: near-zero variance")
        continue

    rf_emb = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf_emb.fit(X_emb, y_emb)

    explainer_emb = shap.TreeExplainer(rf_emb)
    shap_values_emb = explainer_emb.shap_values(X_emb)

    dim_importance = np.abs(shap_values_emb).mean(axis=0)
    all_dim_importances.append(dim_importance)

    for feat, val in zip(embedding_input_cols, dim_importance):
        per_dim_rows.append({
            "GraphEmb_dim": target_col,
            "Feature": feat,
            "MeanAbsSHAP": val
        })

if len(all_dim_importances) == 0:
    raise ValueError("No GraphEmb dimensions had usable variance for embedding SHAP.")

mean_embedding_importance = np.mean(np.vstack(all_dim_importances), axis=0)

embedding_shap_importance = pd.DataFrame({
    "Feature": embedding_input_cols,
    "MeanAbsSHAP": mean_embedding_importance
}).sort_values("MeanAbsSHAP", ascending=False)

embedding_shap_importance.to_csv(
    os.path.join(output_dir, "embedding_shap_importance_graphEmbALLdims.csv"),
    index=False
)

pd.DataFrame(per_dim_rows).to_csv(
    os.path.join(output_dir, "embedding_shap_importance_per_graphEmb_dim.csv"),
    index=False
)

save_importance_barplot(
    embedding_shap_importance,
    os.path.join(output_dir, "Figure_5_embedding_SHAP_bar.png"),
    title="Embedding SHAP importance (Graph embedding, all dimensions)",
    top_n=len(embedding_input_cols)
)

print("Saved embedding SHAP outputs.")

###################### CONTRASTIVE SHAP #########################

graph_emb_cols = [c for c in final_embeddings_df.columns if c.startswith("GraphEmb_")]
final_embeddings_df["GraphEmb_norm"] = np.linalg.norm(
    final_embeddings_df[graph_emb_cols].values,
    axis=1
)

contrastive_merge_df = final_embeddings_df.merge(
    original_feature_df,
    on="Subject_ID",
    how="left"
)

X_con = contrastive_merge_df[embedding_input_cols].copy()
y_con = contrastive_merge_df["GraphEmb_norm"].values

valid_mask_con = ~(X_con.isna().any(axis=1) | pd.isna(y_con))
X_con = X_con.loc[valid_mask_con].reset_index(drop=True)
y_con = y_con[valid_mask_con]

rf_con = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_con.fit(X_con, y_con)

explainer_con = shap.TreeExplainer(rf_con)
shap_values_con = explainer_con.shap_values(X_con)

contrastive_shap_importance = pd.DataFrame({
    "Feature": embedding_input_cols,
    "MeanAbsSHAP": np.abs(shap_values_con).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

contrastive_shap_importance.to_csv(
    os.path.join(output_dir, "contrastive_shap_importance_graphEmbNorm.csv"),
    index=False
)

save_shap_bar_plot(
    shap_values_con,
    X_con,
    os.path.join(output_dir, "Figure_6_contrastive_SHAP_bar.png"),
    title="Contrastive SHAP importance (graph embedding norm)"
)

print("Saved contrastive SHAP outputs.")

###################### INTEGRATED SHAP MATRIX #########################

pred_map = predictive_shap_importance.set_index("Feature")["MeanAbsSHAP"].to_dict()
emb_map = embedding_shap_importance.set_index("Feature")["MeanAbsSHAP"].to_dict()
con_map = contrastive_shap_importance.set_index("Feature")["MeanAbsSHAP"].to_dict()

integration_features = embedding_input_cols

integrated_rows = []
for feat in integration_features:
    integrated_rows.append({
        "Feature": feat,
        "Predictive_SHAP": np.nan,
        "Embedding_SHAP": emb_map.get(feat, np.nan),
        "Contrastive_SHAP": con_map.get(feat, np.nan)
    })

integrated_shap_df = pd.DataFrame(integrated_rows)

predictive_summary_value = predictive_shap_importance["MeanAbsSHAP"].mean()
integrated_shap_df["Predictive_SHAP_global_mean_fused"] = predictive_summary_value

integrated_shap_df["Embedding_rank"] = integrated_shap_df["Embedding_SHAP"].rank(ascending=False, method="average")
integrated_shap_df["Contrastive_rank"] = integrated_shap_df["Contrastive_SHAP"].rank(ascending=False, method="average")
integrated_shap_df["Integrated_score"] = integrated_shap_df[
    ["Embedding_SHAP", "Contrastive_SHAP"]
].mean(axis=1)

integrated_shap_df = integrated_shap_df.sort_values("Integrated_score", ascending=False)

integrated_shap_df.to_csv(
    os.path.join(output_dir, "integrated_SHAP_importance_matrix.csv"),
    index=False
)

plt.figure(figsize=(10, max(6, len(integration_features) * 0.35)))
heatmap_df = integrated_shap_df.set_index("Feature")[
    ["Embedding_SHAP", "Contrastive_SHAP", "Integrated_score"]
]
sns.heatmap(heatmap_df, cmap="viridis", annot=False)
plt.title("Integrated SHAP Importance Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "Figure_7_integrated_SHAP_importance_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved integrated SHAP matrix.")

# NOTE:
# The rest of your downstream plotting/statistics blocks can stay the same as in your original script.
# I stopped here because the key requested fix was the SHAP embedding section and its saved plots.
# If you want, I can also append the remaining unchanged sections verbatim below this point.
# ====================== DOMAIN + EDGE SHAP PLOTS ======================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# =========================================================
# 0) BASIC CHECKS
# =========================================================
def check_shapes(df_fused, df_orig, pred_shap, emb_shap, con_shap):
    assert df_fused.shape[0] == df_orig.shape[0], \
        "df_fused and df_orig must have same number of subjects"
    assert pred_shap.shape[0] == df_fused.shape[0], \
        "pred_shap rows must match df_fused rows"
    assert pred_shap.shape[1] == df_fused.shape[1], \
        "pred_shap columns must match fused features"
    assert emb_shap.shape[0] == df_orig.shape[0], \
        "emb_shap rows must match df_orig rows"
    assert con_shap.shape[0] == df_orig.shape[0], \
        "con_shap rows must match df_orig rows"
    assert emb_shap.shape[1] == df_orig.shape[1], \
        "emb_shap columns must match original features"
    assert con_shap.shape[1] == df_orig.shape[1], \
        "con_shap columns must match original features"


# =========================================================
# 1) SUMMARY IMPORTANCE PER SHAP SET
# =========================================================
def mean_abs_shap(shap_array, feature_names):
    vals = np.mean(np.abs(shap_array), axis=0)
    out = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": vals
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    return out


# =========================================================
# 2) CORRELATION-BASED BACKMAPPING
# =========================================================
def safe_spearman(x, y):
    # Returns abs Spearman correlation, NaN-safe
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    r, _ = spearmanr(x[ok], y[ok])
    return np.abs(r)


def compute_feature_latent_mapping(df_orig, df_fused):
    """
    Returns matrix [n_orig x n_fused] of absolute Spearman correlations
    between original features and fused latent features.
    """
    orig_names = df_orig.columns.tolist()
    fused_names = df_fused.columns.tolist()

    mapping = np.zeros((len(orig_names), len(fused_names)), dtype=float)

    for j, feat in enumerate(orig_names):
        x = df_orig[feat].values.astype(float)
        for k, fused in enumerate(fused_names):
            y = df_fused[fused].values.astype(float)
            mapping[j, k] = safe_spearman(x, y)

    mapping_df = pd.DataFrame(mapping, index=orig_names, columns=fused_names)
    return mapping_df


# =========================================================
# 3) BACKMAPPED PREDICTIVE IMPORTANCE
# =========================================================
def compute_backmapped_predictive_importance(pred_shap, df_fused, mapping_df,
                                             top_k_latents=None,
                                             normalize_latent_importance=True):
    """
    pred_shap: [n_subjects x n_fused]
    mapping_df: [n_orig x n_fused], e.g. abs corr(orig, fused)

    Returns:
      backmap_df with score per original feature
      latent_importance_df with predictive importance per fused dim
    """
    fused_names = df_fused.columns.tolist()

    latent_importance = np.mean(np.abs(pred_shap), axis=0)
    latent_importance_df = pd.DataFrame({
        "FusedFeature": fused_names,
        "PredictiveMeanAbsSHAP": latent_importance
    }).sort_values("PredictiveMeanAbsSHAP", ascending=False).reset_index(drop=True)

    if top_k_latents is not None:
        keep = latent_importance_df["FusedFeature"].iloc[:top_k_latents].tolist()
        latent_mask = np.array([f in keep for f in fused_names])
    else:
        latent_mask = np.ones(len(fused_names), dtype=bool)

    latent_weights = latent_importance.copy()
    latent_weights[~latent_mask] = 0.0

    if normalize_latent_importance and latent_weights.sum() > 0:
        latent_weights = latent_weights / latent_weights.sum()

    # mapping_df.values shape = [n_orig x n_fused]
    backmap_scores = mapping_df.values @ latent_weights

    backmap_df = pd.DataFrame({
        "Feature": mapping_df.index.tolist(),
        "PredictiveBackmappedScore": backmap_scores
    }).sort_values("PredictiveBackmappedScore", ascending=False).reset_index(drop=True)

    return backmap_df, latent_importance_df


# =========================================================
# 4) MERGE ALL THREE VIEWS
# =========================================================
def merge_three_views(backmap_df, emb_df, con_df):
    emb_df2 = emb_df.rename(columns={"MeanAbsSHAP": "EmbeddingMeanAbsSHAP"})
    con_df2 = con_df.rename(columns={"MeanAbsSHAP": "ContrastiveMeanAbsSHAP"})

    merged = backmap_df.merge(emb_df2, on="Feature", how="outer")
    merged = merged.merge(con_df2, on="Feature", how="outer")

    # Fill missing with zero if any
    for c in ["PredictiveBackmappedScore", "EmbeddingMeanAbsSHAP", "ContrastiveMeanAbsSHAP"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    # rank columns
    merged["Rank_PredictiveBackmapped"] = merged["PredictiveBackmappedScore"].rank(
        ascending=False, method="min"
    )
    merged["Rank_Embedding"] = merged["EmbeddingMeanAbsSHAP"].rank(
        ascending=False, method="min"
    )
    merged["Rank_Contrastive"] = merged["ContrastiveMeanAbsSHAP"].rank(
        ascending=False, method="min"
    )

    # Consensus score: simple z-score average
    for c in ["PredictiveBackmappedScore", "EmbeddingMeanAbsSHAP", "ContrastiveMeanAbsSHAP"]:
        mu = merged[c].mean()
        sd = merged[c].std(ddof=0)
        if sd == 0:
            merged[c + "_z"] = 0
        else:
            merged[c + "_z"] = (merged[c] - mu) / sd

    merged["ConsensusScore"] = (
        merged["PredictiveBackmappedScore_z"] +
        merged["EmbeddingMeanAbsSHAP_z"] +
        merged["ContrastiveMeanAbsSHAP_z"]
    ) / 3.0

    merged = merged.sort_values("ConsensusScore", ascending=False).reset_index(drop=True)
    return merged


# =========================================================
# 5) OVERLAP / RECURRENCE ANALYSIS
# =========================================================
def top_feature_overlap(backmap_df, emb_df, con_df, top_n=20):
    top_pred = set(backmap_df["Feature"].head(top_n))
    top_emb = set(emb_df["Feature"].head(top_n))
    top_con = set(con_df["Feature"].head(top_n))

    triple_overlap = sorted(list(top_pred & top_emb & top_con))
    pred_emb_overlap = sorted(list(top_pred & top_emb))
    pred_con_overlap = sorted(list(top_pred & top_con))
    emb_con_overlap = sorted(list(top_emb & top_con))

    overlap_dict = {
        "top_n": top_n,
        "n_pred": len(top_pred),
        "n_emb": len(top_emb),
        "n_con": len(top_con),
        "n_pred_emb": len(pred_emb_overlap),
        "n_pred_con": len(pred_con_overlap),
        "n_emb_con": len(emb_con_overlap),
        "n_triple": len(triple_overlap),
        "pred_emb_overlap_features": pred_emb_overlap,
        "pred_con_overlap_features": pred_con_overlap,
        "emb_con_overlap_features": emb_con_overlap,
        "triple_overlap_features": triple_overlap,
    }
    return overlap_dict


# =========================================================
# 6) OPTIONAL FEATURE GROUPING
# =========================================================
def assign_feature_group(feature_name):
    """
    Edit these rules to match your naming conventions.
    """
    f = feature_name.lower()

    if any(x in f for x in ["age", "sex", "apoe", "bmi", "education"]):
        return "demographics"
    elif "pc" in f:
        return "pcs"
    elif any(x in f for x in ["hc_", "hipp", "amyg", "ctx", "volume", "fa", "md", "rd", "ad"]):
        return "regions"
    elif any(x in f for x in ["edge", "conn", "--", "to_", "_to_"]):
        return "connections"
    else:
        return "other"


def add_feature_groups(df, feature_col="Feature"):
    df = df.copy()
    df["Group"] = df[feature_col].apply(assign_feature_group)
    return df


# =========================================================
# 7) PLOTS
# =========================================================
def plot_top_features(df, score_col, title, outpath, top_n=20):
    sub = df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, max(6, top_n * 0.28)))
    plt.barh(sub["Feature"], sub[score_col])
    plt.xlabel(score_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_summary(merged_df, outpath):
    tmp = add_feature_groups(merged_df)

    grp = tmp.groupby("Group")[[
        "PredictiveBackmappedScore",
        "EmbeddingMeanAbsSHAP",
        "ContrastiveMeanAbsSHAP",
        "ConsensusScore"
    ]].mean().reset_index()

    grp = grp.sort_values("ConsensusScore", ascending=False)

    x = np.arange(len(grp))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, grp["PredictiveBackmappedScore"], width, label="Predictive-backmapped")
    plt.bar(x,         grp["EmbeddingMeanAbsSHAP"],    width, label="Embedding")
    plt.bar(x + width, grp["ContrastiveMeanAbsSHAP"],  width, label="Contrastive")
    plt.xticks(x, grp["Group"], rotation=30, ha="right")
    plt.ylabel("Mean importance")
    plt.title("Importance by feature group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 8) MAIN PIPELINE
# =========================================================
def run_shap_backmapping_pipeline(
    df_fused,
    df_orig,
    pred_shap,
    emb_shap,
    con_shap,
    outdir,
    top_k_latents=20,
    top_n_overlap=20
):
    check_shapes(df_fused, df_orig, pred_shap, emb_shap, con_shap)

    fused_names = df_fused.columns.tolist()
    orig_names = df_orig.columns.tolist()

    # 1) SHAP summaries
    emb_df = mean_abs_shap(emb_shap, orig_names)
    con_df = mean_abs_shap(con_shap, orig_names)

    # 2) Mapping original <-> latent
    mapping_df = compute_feature_latent_mapping(df_orig, df_fused)
    mapping_df.to_csv(os.path.join(outdir, "orig_to_fused_abs_spearman_mapping.csv"))

    # 3) Backmapped predictive importance
    backmap_df, latent_importance_df = compute_backmapped_predictive_importance(
        pred_shap=pred_shap,
        df_fused=df_fused,
        mapping_df=mapping_df,
        top_k_latents=top_k_latents,
        normalize_latent_importance=True
    )

    # 4) Merge all three views
    merged_df = merge_three_views(backmap_df, emb_df, con_df)
    merged_df = add_feature_groups(merged_df)

    # 5) Overlap
    overlap = top_feature_overlap(backmap_df, emb_df, con_df, top_n=top_n_overlap)
    overlap_df = pd.DataFrame({
        "TripleOverlapFeatures": pd.Series(overlap["triple_overlap_features"])
    })

    # 6) Save tables
    latent_importance_df.to_csv(os.path.join(outdir, "predictive_latent_importance.csv"), index=False)
    backmap_df.to_csv(os.path.join(outdir, "predictive_backmapped_importance.csv"), index=False)
    emb_df.to_csv(os.path.join(outdir, "embedding_importance.csv"), index=False)
    con_df.to_csv(os.path.join(outdir, "contrastive_importance.csv"), index=False)
    merged_df.to_csv(os.path.join(outdir, "merged_three_view_importance.csv"), index=False)
    overlap_df.to_csv(os.path.join(outdir, f"triple_overlap_top{top_n_overlap}.csv"), index=False)

    # 7) Save text summary
    with open(os.path.join(outdir, "overlap_summary.txt"), "w") as f:
        f.write(f"Top N = {overlap['top_n']}\n")
        f.write(f"Predictive-backmapped ∩ Embedding: {overlap['n_pred_emb']}\n")
        f.write(f"Predictive-backmapped ∩ Contrastive: {overlap['n_pred_con']}\n")
        f.write(f"Embedding ∩ Contrastive: {overlap['n_emb_con']}\n")
        f.write(f"Triple overlap: {overlap['n_triple']}\n\n")

        f.write("Triple-overlap features:\n")
        for feat in overlap["triple_overlap_features"]:
            f.write(f"- {feat}\n")

    # 8) Plots
    plot_top_features(
        latent_importance_df.rename(columns={"FusedFeature": "Feature",
                                             "PredictiveMeanAbsSHAP": "Score"}),
        score_col="Score",
        title=f"Top predictive latent fused dimensions (top {min(top_k_latents, len(fused_names))})",
        outpath=os.path.join(outdir, "top_predictive_latent_dims.png"),
        top_n=min(top_k_latents, len(fused_names))
    )

    plot_top_features(
        backmap_df,
        score_col="PredictiveBackmappedScore",
        title="Top original metrics from predictive latent SHAP backmapping",
        outpath=os.path.join(outdir, "top_predictive_backmapped_features.png"),
        top_n=20
    )

    plot_top_features(
        emb_df,
        score_col="MeanAbsSHAP",
        title="Top embedding SHAP features",
        outpath=os.path.join(outdir, "top_embedding_features.png"),
        top_n=20
    )

    plot_top_features(
        con_df,
        score_col="MeanAbsSHAP",
        title="Top contrastive SHAP features",
        outpath=os.path.join(outdir, "top_contrastive_features.png"),
        top_n=20
    )

    plot_top_features(
        merged_df,
        score_col="ConsensusScore",
        title="Consensus features across predictive-backmapped, embedding, contrastive",
        outpath=os.path.join(outdir, "top_consensus_features.png"),
        top_n=20
    )

    plot_group_summary(
        merged_df,
        outpath=os.path.join(outdir, "group_summary_three_views.png")
    )

    print("\nSaved outputs to:", outdir)
    print("\nTop latent predictive dimensions:")
    print(latent_importance_df.head(10))

    print("\nTop predictive-backmapped original features:")
    print(backmap_df.head(10))

    print("\nTop embedding SHAP features:")
    print(emb_df.head(10))

    print("\nTop contrastive SHAP features:")
    print(con_df.head(10))

    print("\nTop consensus features:")
    print(merged_df[[
        "Feature",
        "Group",
        "PredictiveBackmappedScore",
        "EmbeddingMeanAbsSHAP",
        "ContrastiveMeanAbsSHAP",
        "ConsensusScore"
    ]].head(20))

    print("\nTriple overlap features:")
    print(overlap["triple_overlap_features"])

    return {
        "mapping_df": mapping_df,
        "latent_importance_df": latent_importance_df,
        "backmap_df": backmap_df,
        "emb_df": emb_df,
        "con_df": con_df,
        "merged_df": merged_df,
        "overlap": overlap
    }

# ====================== APPEND BELOW: SHAP BARPLOTS + BEESWARMS + EDGE/REGION PLOTS ======================

# ====================== APPEND BELOW: SHAP BARPLOTS + BEESWARMS + EDGE/REGION PLOTS ======================

import warnings

# ------------------------------------------------------------------
# 1) ATLAS NAME MAP
# ------------------------------------------------------------------
extra_plot_dir = os.path.join(output_dir, "extra_shap_plots")
os.makedirs(extra_plot_dir, exist_ok=True)

atlas_name_map_path = os.path.join(WORK, "ines/data/atlas/IITmean_RPI_index.xlsx")

def clean_region_name(name):
    name = str(name).strip().replace('"', "")
    if name.startswith("ctx-lh-"):
        name = name.replace("ctx-lh-", "Left ")
    elif name.startswith("ctx-rh-"):
        name = name.replace("ctx-rh-", "Right ")
    name = name.replace("Left-", "Left ")
    name = name.replace("Right-", "Right ")
    name = name.replace("-Proper", "")
    name = name.replace("-area", "")
    name = name.replace("-", " ")
    name = " ".join(name.split())
    return name.strip()

# leer atlas
df_atlas = pd.read_excel(atlas_name_map_path)
df_atlas.columns = [str(c).strip() for c in df_atlas.columns]

print("\n=== ATLAS DEBUG ===")
print("Atlas path:", atlas_name_map_path)
print("Atlas columns:", df_atlas.columns.tolist())

# usar SOLO estas columnas
df_atlas = df_atlas[["index2", "Structure"]].copy()
df_atlas["index2"] = pd.to_numeric(df_atlas["index2"], errors="coerce")
df_atlas = df_atlas.dropna(subset=["index2", "Structure"])
df_atlas["index2"] = df_atlas["index2"].astype(int)

# construir mapa 1..84 -> nombre
ROI_NAME_MAP = {i: f"Region {i}" for i in range(1, 85)}

for _, row in df_atlas.iterrows():
    roi_idx = int(row["index2"])
    if 1 <= roi_idx <= 84:
        ROI_NAME_MAP[roi_idx] = clean_region_name(row["Structure"])

print("\n=== FINAL ROI_NAME_MAP CHECK ===")
for i in range(1, 15):
    print(f"{i} -> {ROI_NAME_MAP[i]}")

pd.DataFrame({
    "ROI_idx": list(range(1, 85)),
    "Region_name": [ROI_NAME_MAP[i] for i in range(1, 85)]
}).to_csv(
    os.path.join(extra_plot_dir, "ROI_index_to_region_name.csv"),
    index=False
)

# ------------------------------------------------------------------
# 2) GENERAL HELPERS
# ------------------------------------------------------------------
def savefig_show(outpath, dpi=300):
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()

def classify_feature(feature_name):
    f = str(feature_name).lower()
    if f in ["systolic", "diastolic", "sex_encoded", "genotype_encoded"]:
        return "demographics"
    elif f in ["clustering_coeff", "path_length"]:
        return "graph_metrics"
    elif f.startswith("pc"):
        return "pcs"
    elif any(x in f for x in ["roi_", "region ", "hipp", "amyg", "ctx", "volume", "fa", "md"]):
        return "regions"
    elif any(x in f for x in ["edge", "conn", "--", "_to_", " to "]):
        return "connections"
    else:
        return "other"

def pretty_input_feature_name(f):
    rename_map = {
        "Systolic": "Systolic BP",
        "Diastolic": "Diastolic BP",
        "sex_encoded": "Sex",
        "genotype_encoded": "APOE genotype",
        "Clustering_Coeff": "Clustering coefficient",
        "Path_Length": "Path length",
    }

    f_str = str(f)

    if f_str.startswith("ROI_"):
        try:
            roi_num = int(f_str.replace("ROI_", ""))
            return clean_region_name(ROI_NAME_MAP.get(roi_num, f"Region {roi_num}"))
        except Exception:
            return f_str

    return rename_map.get(f_str, f_str)

def signed_spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    r, _ = spearmanr(x[ok], y[ok])
    return r

def build_signed_mapping(df_orig_num, df_fused_num):
    orig_names = df_orig_num.columns.tolist()
    fused_names = df_fused_num.columns.tolist()

    arr = np.zeros((len(orig_names), len(fused_names)), dtype=float)

    for j, feat in enumerate(orig_names):
        x = df_orig_num[feat].values.astype(float)
        for k, fused in enumerate(fused_names):
            y = df_fused_num[fused].values.astype(float)
            r = signed_spearman(x, y)
            arr[j, k] = 0.0 if pd.isna(r) else r

    return pd.DataFrame(arr, index=orig_names, columns=fused_names)

def make_barplot(df, score_col, title, outpath, top_n=20):
    sub = df.head(top_n).copy().iloc[::-1]
    sub["PrettyFeature"] = sub["Feature"].map(pretty_input_feature_name)

    plt.figure(figsize=(9, max(5, 0.38 * len(sub))))
    plt.barh(sub["PrettyFeature"], sub[score_col])
    plt.xlabel(score_col)
    plt.ylabel("")
    plt.title(title)
    savefig_show(outpath)

def make_grouped_barplot(merged_df, outpath, title="Mean importance by feature group"):
    tmp = merged_df.copy()
    tmp["Group"] = tmp["Feature"].apply(classify_feature)

    grp = tmp.groupby("Group")[[
        "PredictiveBackmappedScore",
        "EmbeddingMeanSHAP_signedAbsMean",
        "ContrastiveMeanSHAP_signedAbsMean"
    ]].mean().reset_index()

    desired_order = ["demographics", "graph_metrics", "pcs", "regions", "connections", "other"]
    grp["order"] = grp["Group"].apply(lambda x: desired_order.index(x) if x in desired_order else 999)
    grp = grp.sort_values("order")

    x = np.arange(len(grp))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, grp["PredictiveBackmappedScore"], width, label="Predictive-backmapped")
    plt.bar(x,         grp["EmbeddingMeanSHAP_signedAbsMean"], width, label="Embedding")
    plt.bar(x + width, grp["ContrastiveMeanSHAP_signedAbsMean"], width, label="Contrastive")

    plt.xticks(x, grp["Group"], rotation=25, ha="right")
    plt.ylabel("Mean importance")
    plt.title(title)
    plt.legend()
    savefig_show(outpath)

def make_beeswarm(shap_matrix, X_df, title, outpath, max_display=20):
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        X_df,
        show=False,
        max_display=max_display
    )
    plt.title(title)
    savefig_show(outpath)

def make_barplot_by_group(df, score_col, group_name, outpath, title=None, top_n=20):
    tmp = df.copy()
    tmp["Group"] = tmp["Feature"].apply(classify_feature)
    tmp = tmp[tmp["Group"] == group_name].copy()

    if tmp.empty:
        print(f"[SKIP] No features found for group '{group_name}' in {score_col}")
        return

    tmp = tmp.sort_values(score_col, ascending=False).head(top_n)
    make_barplot(
        tmp.rename(columns={score_col: "PlotScore"}),
        score_col="PlotScore",
        title=title if title is not None else f"{group_name} - {score_col}",
        outpath=outpath,
        top_n=len(tmp)
    )

def subset_X_by_group(X_df, group_name):
    cols = [c for c in X_df.columns if classify_feature(c) == group_name]
    return X_df[cols].copy(), cols

def subset_shap_by_group(shap_mat, X_df, group_name):
    cols = [c for c in X_df.columns if classify_feature(c) == group_name]
    idx = [X_df.columns.get_loc(c) for c in cols]
    if len(idx) == 0:
        return None, None
    return shap_mat[:, idx], X_df[cols].copy()


# ------------------------------------------------------------------
# 3) BUILD PREDICTIVE BACKMAPPED SUBJECT-LEVEL MATRIX
# ------------------------------------------------------------------
predictive_merge_df = final_embeddings_df.merge(
    original_feature_df,
    on="Subject_ID",
    how="left"
)

predictive_input_cols_original = embedding_input_cols.copy()
predictive_fused_cols = predictive_feature_cols.copy()

valid_mask_pred_backmap = ~(
    predictive_merge_df[predictive_input_cols_original].isna().any(axis=1) |
    predictive_merge_df[predictive_fused_cols].isna().any(axis=1)
)

pred_backmap_df_valid = predictive_merge_df.loc[valid_mask_pred_backmap].reset_index(drop=True)

X_fused_valid = pred_backmap_df_valid[predictive_fused_cols].copy()
X_orig_valid = pred_backmap_df_valid[predictive_input_cols_original].copy()

shap_values_pred_valid = np.array(shap_values_pred)[valid_mask_pred_backmap.values, :]

signed_mapping_df = build_signed_mapping(X_orig_valid, X_fused_valid)

pred_backmapped_subject_mat = shap_values_pred_valid @ signed_mapping_df.values.T

pred_backmapped_subject_df = pd.DataFrame(
    pred_backmapped_subject_mat,
    columns=predictive_input_cols_original
)

pred_backmapped_importance_df = pd.DataFrame({
    "Feature": predictive_input_cols_original,
    "PredictiveBackmappedScore": np.mean(np.abs(pred_backmapped_subject_mat), axis=0)
}).sort_values("PredictiveBackmappedScore", ascending=False).reset_index(drop=True)

pred_backmapped_importance_df.to_csv(
    os.path.join(extra_plot_dir, "predictive_backmapped_subjectlevel_importance.csv"),
    index=False
)

pred_backmapped_subject_df.to_csv(
    os.path.join(extra_plot_dir, "predictive_backmapped_subjectlevel_matrix.csv"),
    index=False
)


# ------------------------------------------------------------------
# 4) RECOMPUTE EMBEDDING SHAP AS SUBJECT-LEVEL MATRIX
# ------------------------------------------------------------------
print("\n=== Recomputing embedding SHAP subject-level matrix for beeswarm plots ===")

embedding_signed_mats = []

for target_col in Y_emb.columns:
    y_emb_dim = Y_emb[target_col].values

    if np.nanstd(y_emb_dim) < 1e-8:
        continue

    rf_emb_dim = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf_emb_dim.fit(X_emb, y_emb_dim)

    explainer_emb_dim = shap.TreeExplainer(rf_emb_dim)
    shap_values_emb_dim = explainer_emb_dim.shap_values(X_emb)

    embedding_signed_mats.append(shap_values_emb_dim)

if len(embedding_signed_mats) == 0:
    raise ValueError("No usable embedding dimensions to build subject-level embedding SHAP matrix.")

embedding_shap_subject_mat = np.mean(np.stack(embedding_signed_mats, axis=0), axis=0)

embedding_shap_subject_df = pd.DataFrame(
    embedding_shap_subject_mat,
    columns=X_emb.columns
)

embedding_shap_subject_df.to_csv(
    os.path.join(extra_plot_dir, "embedding_subjectlevel_shap_matrix.csv"),
    index=False
)

embedding_shap_importance_subject_df = pd.DataFrame({
    "Feature": X_emb.columns,
    "EmbeddingMeanSHAP_signedAbsMean": np.mean(np.abs(embedding_shap_subject_mat), axis=0)
}).sort_values("EmbeddingMeanSHAP_signedAbsMean", ascending=False).reset_index(drop=True)

embedding_shap_importance_subject_df.to_csv(
    os.path.join(extra_plot_dir, "embedding_subjectlevel_importance.csv"),
    index=False
)


# ------------------------------------------------------------------
# 5) CONTRASTIVE SUBJECT-LEVEL MATRIX
# ------------------------------------------------------------------
contrastive_shap_subject_mat = np.array(shap_values_con)

contrastive_shap_subject_df = pd.DataFrame(
    contrastive_shap_subject_mat,
    columns=X_con.columns
)

contrastive_shap_subject_df.to_csv(
    os.path.join(extra_plot_dir, "contrastive_subjectlevel_shap_matrix.csv"),
    index=False
)

contrastive_shap_importance_subject_df = pd.DataFrame({
    "Feature": X_con.columns,
    "ContrastiveMeanSHAP_signedAbsMean": np.mean(np.abs(contrastive_shap_subject_mat), axis=0)
}).sort_values("ContrastiveMeanSHAP_signedAbsMean", ascending=False).reset_index(drop=True)

contrastive_shap_importance_subject_df.to_csv(
    os.path.join(extra_plot_dir, "contrastive_subjectlevel_importance.csv"),
    index=False
)


# ------------------------------------------------------------------
# 6) MERGE THE 3 COMPARABLE VIEWS
# ------------------------------------------------------------------
three_view_df = pred_backmapped_importance_df.merge(
    embedding_shap_importance_subject_df,
    on="Feature",
    how="outer"
).merge(
    contrastive_shap_importance_subject_df,
    on="Feature",
    how="outer"
)

for c in ["PredictiveBackmappedScore", "EmbeddingMeanSHAP_signedAbsMean", "ContrastiveMeanSHAP_signedAbsMean"]:
    three_view_df[c] = three_view_df[c].fillna(0.0)

three_view_df["Group"] = three_view_df["Feature"].apply(classify_feature)

for c in ["PredictiveBackmappedScore", "EmbeddingMeanSHAP_signedAbsMean", "ContrastiveMeanSHAP_signedAbsMean"]:
    mu = three_view_df[c].mean()
    sd = three_view_df[c].std(ddof=0)
    if sd == 0:
        three_view_df[c + "_z"] = 0.0
    else:
        three_view_df[c + "_z"] = (three_view_df[c] - mu) / sd

three_view_df["ConsensusScore"] = (
    three_view_df["PredictiveBackmappedScore_z"] +
    three_view_df["EmbeddingMeanSHAP_signedAbsMean_z"] +
    three_view_df["ContrastiveMeanSHAP_signedAbsMean_z"]
) / 3.0

three_view_df = three_view_df.sort_values("ConsensusScore", ascending=False).reset_index(drop=True)

three_view_df.to_csv(
    os.path.join(extra_plot_dir, "three_view_comparable_importance.csv"),
    index=False
)


# ------------------------------------------------------------------
# 7) GLOBAL BAR PLOTS (ALL COMPARABLE FEATURES)
# ------------------------------------------------------------------
make_barplot(
    pred_backmapped_importance_df,
    score_col="PredictiveBackmappedScore",
    title="Predictive backmapped SHAP (global bar plot)",
    outpath=os.path.join(extra_plot_dir, "bar_predictive_backmapped_all.png"),
    top_n=len(pred_backmapped_importance_df)
)

make_barplot(
    embedding_shap_importance_subject_df.rename(
        columns={"EmbeddingMeanSHAP_signedAbsMean": "ScoreTmp"}
    ),
    score_col="ScoreTmp",
    title="Embedding SHAP (global bar plot)",
    outpath=os.path.join(extra_plot_dir, "bar_embedding_all.png"),
    top_n=len(embedding_shap_importance_subject_df)
)

make_barplot(
    contrastive_shap_importance_subject_df.rename(
        columns={"ContrastiveMeanSHAP_signedAbsMean": "ScoreTmp"}
    ),
    score_col="ScoreTmp",
    title="Contrastive SHAP (global bar plot)",
    outpath=os.path.join(extra_plot_dir, "bar_contrastive_all.png"),
    top_n=len(contrastive_shap_importance_subject_df)
)

make_barplot(
    three_view_df.rename(columns={"ConsensusScore": "ScoreTmp"}),
    score_col="ScoreTmp",
    title="Consensus across predictive-backmapped, embedding and contrastive",
    outpath=os.path.join(extra_plot_dir, "bar_consensus_all.png"),
    top_n=len(three_view_df)
)

make_grouped_barplot(
    three_view_df,
    outpath=os.path.join(extra_plot_dir, "bar_group_summary_three_views.png"),
    title="Mean importance by feature group across the three SHAP views"
)


# ------------------------------------------------------------------
# 8) BAR PLOTS BY GROUP
# ------------------------------------------------------------------
for grp in ["demographics", "graph_metrics", "pcs"]:
    make_barplot_by_group(
        pred_backmapped_importance_df,
        score_col="PredictiveBackmappedScore",
        group_name=grp,
        outpath=os.path.join(extra_plot_dir, f"bar_predictive_backmapped_{grp}.png"),
        title=f"Predictive backmapped SHAP - {grp}"
    )

    make_barplot_by_group(
        embedding_shap_importance_subject_df,
        score_col="EmbeddingMeanSHAP_signedAbsMean",
        group_name=grp,
        outpath=os.path.join(extra_plot_dir, f"bar_embedding_{grp}.png"),
        title=f"Embedding SHAP - {grp}"
    )

    make_barplot_by_group(
        contrastive_shap_importance_subject_df,
        score_col="ContrastiveMeanSHAP_signedAbsMean",
        group_name=grp,
        outpath=os.path.join(extra_plot_dir, f"bar_contrastive_{grp}.png"),
        title=f"Contrastive SHAP - {grp}"
    )


# ------------------------------------------------------------------
# 9) BEESWARM PLOTS (ALL COMPARABLE FEATURES)
# ------------------------------------------------------------------
make_beeswarm(
    pred_backmapped_subject_mat,
    X_orig_valid.rename(columns=pretty_input_feature_name),
    title="Predictive backmapped SHAP beeswarm",
    outpath=os.path.join(extra_plot_dir, "beeswarm_predictive_backmapped_all.png"),
    max_display=len(X_orig_valid.columns)
)

make_beeswarm(
    embedding_shap_subject_mat,
    X_emb.rename(columns=pretty_input_feature_name),
    title="Embedding SHAP beeswarm",
    outpath=os.path.join(extra_plot_dir, "beeswarm_embedding_all.png"),
    max_display=len(X_emb.columns)
)

make_beeswarm(
    contrastive_shap_subject_mat,
    X_con.rename(columns=pretty_input_feature_name),
    title="Contrastive SHAP beeswarm",
    outpath=os.path.join(extra_plot_dir, "beeswarm_contrastive_all.png"),
    max_display=len(X_con.columns)
)


# ------------------------------------------------------------------
# 10) BEESWARM PLOTS BY GROUP
# ------------------------------------------------------------------
for grp in ["demographics", "graph_metrics", "pcs"]:
    shap_sub, X_sub = subset_shap_by_group(pred_backmapped_subject_mat, X_orig_valid, grp)
    if shap_sub is not None and X_sub.shape[1] > 0:
        make_beeswarm(
            shap_sub,
            X_sub.rename(columns=pretty_input_feature_name),
            title=f"Predictive backmapped SHAP beeswarm - {grp}",
            outpath=os.path.join(extra_plot_dir, f"beeswarm_predictive_backmapped_{grp}.png"),
            max_display=X_sub.shape[1]
        )

    shap_sub, X_sub = subset_shap_by_group(embedding_shap_subject_mat, X_emb, grp)
    if shap_sub is not None and X_sub.shape[1] > 0:
        make_beeswarm(
            shap_sub,
            X_sub.rename(columns=pretty_input_feature_name),
            title=f"Embedding SHAP beeswarm - {grp}",
            outpath=os.path.join(extra_plot_dir, f"beeswarm_embedding_{grp}.png"),
            max_display=X_sub.shape[1]
        )

    shap_sub, X_sub = subset_shap_by_group(contrastive_shap_subject_mat, X_con, grp)
    if shap_sub is not None and X_sub.shape[1] > 0:
        make_beeswarm(
            shap_sub,
            X_sub.rename(columns=pretty_input_feature_name),
            title=f"Contrastive SHAP beeswarm - {grp}",
            outpath=os.path.join(extra_plot_dir, f"beeswarm_contrastive_{grp}.png"),
            max_display=X_sub.shape[1]
        )


# ------------------------------------------------------------------
# 11) TOP CONNECTIONS FROM EDGE-LEVEL SHAP
# ------------------------------------------------------------------
def guess_edge_pairs_from_cols(edge_cols, n_nodes=84):
    """
    Tries to infer node pairs from your SHAP column names.
    Supports:
      - upper-triangular order fallback
      - names containing two integers, e.g. edge_3_18 or 3_18
    Returns a list of tuples (i, j) in 1-based indexing.
    """
    pairs = []

    parsed_ok = True
    for c in edge_cols:
        nums = re.findall(r"\d+", str(c))
        if len(nums) >= 2:
            i = int(nums[0])
            j = int(nums[1])

            if i == 0 or j == 0:
                i += 1
                j += 1

            if i == j:
                parsed_ok = False
                break
            pairs.append((i, j))
        else:
            parsed_ok = False
            break

    if parsed_ok and len(pairs) == len(edge_cols):
        return pairs

    ui, uj = np.triu_indices(n_nodes, k=1)
    pairs = list(zip(ui + 1, uj + 1))
    if len(pairs) != len(edge_cols):
        raise ValueError(
            f"Could not infer edge pairs. Number of edge columns = {len(edge_cols)} "
            f"but upper-triangular for n_nodes={n_nodes} gives {len(pairs)}."
        )
    return pairs

edge_pairs_1based = guess_edge_pairs_from_cols(shap_feature_cols, n_nodes=84)

edge_importance_df = pd.DataFrame({
    "EdgeCol": shap_feature_cols,
    "Node_i": [p[0] for p in edge_pairs_1based],
    "Node_j": [p[1] for p in edge_pairs_1based],
    "Region_i": [clean_region_name(ROI_NAME_MAP[p[0]]) for p in edge_pairs_1based],
    "Region_j": [clean_region_name(ROI_NAME_MAP[p[1]]) for p in edge_pairs_1based],
    "Connection": [
        f"{clean_region_name(ROI_NAME_MAP[p[0]])} ↔ {clean_region_name(ROI_NAME_MAP[p[1]])}"
        for p in edge_pairs_1based
    ],
    "MeanAbsSHAP": np.mean(np.abs(df_shap[shap_feature_cols].values), axis=0)
}).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

edge_importance_df.to_csv(
    os.path.join(extra_plot_dir, "edge_level_top_connections.csv"),
    index=False
)

plt.figure(figsize=(10, max(6, 0.35 * 20)))
sub = edge_importance_df.head(20).iloc[::-1]
plt.barh(sub["Connection"], sub["MeanAbsSHAP"])
plt.xlabel("Mean |edge SHAP|")
plt.ylabel("")
plt.title("Top 20 connections from edge-level SHAP")
savefig_show(os.path.join(extra_plot_dir, "bar_top20_connections_edge_shap.png"))


# ------------------------------------------------------------------
# 12) TOP REGIONS FROM EDGE-LEVEL SHAP
# ------------------------------------------------------------------
region_scores = {roi_idx: 0.0 for roi_idx in range(1, 85)}

for _, row in edge_importance_df.iterrows():
    region_scores[int(row["Node_i"])] += float(row["MeanAbsSHAP"])
    region_scores[int(row["Node_j"])] += float(row["MeanAbsSHAP"])

region_importance_df = pd.DataFrame({
    "ROI_idx": list(region_scores.keys()),
    "Feature": [clean_region_name(ROI_NAME_MAP[k]) for k in region_scores.keys()],
    "MeanAbsSHAP": list(region_scores.values())
}).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

region_importance_df.to_csv(
    os.path.join(extra_plot_dir, "edge_derived_top_regions.csv"),
    index=False
)

plt.figure(figsize=(10, max(6, 0.35 * 20)))
sub = region_importance_df.head(20).iloc[::-1]
plt.barh(sub["Feature"], sub["MeanAbsSHAP"])
plt.xlabel("Summed incident edge |SHAP|")
plt.ylabel("")
plt.title("Top 20 regions from edge-level SHAP")
savefig_show(os.path.join(extra_plot_dir, "bar_top20_regions_edge_shap.png"))


# ------------------------------------------------------------------
# 13) OPTIONAL SUBJECT-LEVEL HEATMAPS FOR TOP REGIONS / CONNECTIONS
# ------------------------------------------------------------------
top_conn_cols = edge_importance_df["EdgeCol"].head(20).tolist()
top_conn_names = edge_importance_df["Connection"].head(20).tolist()

top_conn_mat = df_shap[top_conn_cols].copy()
top_conn_mat.columns = top_conn_names

plt.figure(figsize=(12, 8))
sns.heatmap(top_conn_mat.iloc[:min(50, len(top_conn_mat))].T, cmap="coolwarm", center=0)
plt.title("Top 20 connection SHAP values across first subjects")
plt.xlabel("Subjects")
plt.ylabel("Connections")
savefig_show(os.path.join(extra_plot_dir, "heatmap_top20_connections_subjects.png"))

edge_shap_subject_mat = df_shap[shap_feature_cols].values
region_subject_scores = np.zeros((edge_shap_subject_mat.shape[0], 84), dtype=float)

for e_idx, (i, j) in enumerate(edge_pairs_1based):
    region_subject_scores[:, i - 1] += np.abs(edge_shap_subject_mat[:, e_idx])
    region_subject_scores[:, j - 1] += np.abs(edge_shap_subject_mat[:, e_idx])

region_subject_df = pd.DataFrame(
    region_subject_scores,
    columns=[clean_region_name(ROI_NAME_MAP[i]) for i in range(1, 85)]
)

top_region_names = region_importance_df["Feature"].head(20).tolist()

plt.figure(figsize=(12, 8))
sns.heatmap(region_subject_df[top_region_names].iloc[:min(50, len(region_subject_df))].T, cmap="viridis")
plt.title("Top 20 region scores across first subjects")
plt.xlabel("Subjects")
plt.ylabel("Regions")
savefig_show(os.path.join(extra_plot_dir, "heatmap_top20_regions_subjects.png"))


# ------------------------------------------------------------------
# 14) SIMPLE OVERLAP TABLES FOR THE 3 COMPARABLE SHAP VIEWS
# ------------------------------------------------------------------
top_n_overlap = 10

top_pred_set = set(pred_backmapped_importance_df["Feature"].head(top_n_overlap))
top_emb_set = set(embedding_shap_importance_subject_df["Feature"].head(top_n_overlap))
top_con_set = set(contrastive_shap_importance_subject_df["Feature"].head(top_n_overlap))

triple_overlap = sorted(list(top_pred_set & top_emb_set & top_con_set))

overlap_rows = []
for feat in sorted(list(top_pred_set | top_emb_set | top_con_set)):
    overlap_rows.append({
        "Feature": feat,
        "PrettyFeature": pretty_input_feature_name(feat),
        "InTopPredictiveBackmapped": feat in top_pred_set,
        "InTopEmbedding": feat in top_emb_set,
        "InTopContrastive": feat in top_con_set,
    })

overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(
    os.path.join(extra_plot_dir, f"top{top_n_overlap}_overlap_three_views.csv"),
    index=False
)

with open(os.path.join(extra_plot_dir, f"top{top_n_overlap}_triple_overlap.txt"), "w") as f:
    f.write(f"Top {top_n_overlap} triple-overlap features\n")
    f.write("=" * 50 + "\n")
    for feat in triple_overlap:
        f.write(pretty_input_feature_name(feat) + "\n")

print("\n=== DONE: extra SHAP plots saved in ===")
print(extra_plot_dir)
print("Triple overlap features:", [pretty_input_feature_name(x) for x in triple_overlap])


# ====================== ORDER HEATMAPS BY EXISTING CLUSTERS ======================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1) LOAD FILE WITH CLUSTERS
# ------------------------------------------------------------------
cluster_file = os.path.join(
    WORK,
    "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics_plusCluster.xlsx"
)

df_clusters = pd.read_excel(cluster_file)

print("\n=== CLUSTER FILE DEBUG ===")
print("Cluster file:", cluster_file)
print("Columns found:")
print(df_clusters.columns.tolist())

# ------------------------------------------------------------------
# 2) DEFINE COLUMN NAMES
# ------------------------------------------------------------------
# Change these ONLY if your file uses different names
cluster_id_col = "MRI_Exam"
cluster_col = "Cluster_HC"

if cluster_id_col not in df_clusters.columns:
    raise ValueError(f"Column '{cluster_id_col}' not found in cluster file.")

if cluster_col not in df_clusters.columns:
    raise ValueError(f"Column '{cluster_col}' not found in cluster file.")

# ------------------------------------------------------------------
# 3) FIX IDS AND MERGE WITH SHAP TABLE
# ------------------------------------------------------------------
df_clusters["Subject_ID_fixed"] = (
    df_clusters[cluster_id_col]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

df_shap["Subject_ID_fixed"] = df_shap["Subject_ID"].astype(str).str.zfill(5)

df_shap_clustered = df_shap.merge(
    df_clusters[["Subject_ID_fixed", cluster_col]],
    on="Subject_ID_fixed",
    how="left"
)

print("\n=== MERGE CHECK ===")
print(df_shap_clustered[["Subject_ID", "Subject_ID_fixed", cluster_col]].head())
print("\nCluster counts after merge:")
print(df_shap_clustered[cluster_col].value_counts(dropna=False).sort_index())

# ------------------------------------------------------------------
# 4) OUTPUT DIR
# ------------------------------------------------------------------
cluster_plot_dir = os.path.join(extra_plot_dir, "ordered_by_existing_clusters")
os.makedirs(cluster_plot_dir, exist_ok=True)

# ------------------------------------------------------------------
# 5) PREPARE TOP CONNECTIONS TABLE ORDERED BY CLUSTER
# ------------------------------------------------------------------
top_conn_cols = edge_importance_df["EdgeCol"].head(20).tolist()
top_conn_names = edge_importance_df["Connection"].head(20).tolist()

conn_plot_df = df_shap_clustered[["Subject_ID_fixed", cluster_col] + top_conn_cols].copy()
conn_plot_df = conn_plot_df.dropna(subset=[cluster_col]).copy()
conn_plot_df[cluster_col] = conn_plot_df[cluster_col].astype(int)

conn_plot_df = conn_plot_df.sort_values([cluster_col, "Subject_ID_fixed"]).reset_index(drop=True)

conn_heatmap_mat = conn_plot_df[top_conn_cols].T
conn_heatmap_mat.index = top_conn_names

# boundaries between clusters
cluster_sizes_conn = conn_plot_df[cluster_col].value_counts().sort_index()
boundaries_conn = np.cumsum(cluster_sizes_conn.values)[:-1]

# ------------------------------------------------------------------
# 6) PLOT TOP CONNECTIONS HEATMAP ORDERED BY CLUSTER
# ------------------------------------------------------------------
plt.figure(figsize=(16, 9))
ax = sns.heatmap(conn_heatmap_mat, cmap="coolwarm", center=0)

for b in boundaries_conn:
    ax.vlines(b, *ax.get_ylim(), colors="black", linewidth=2)

# put cluster labels at centers
cluster_centers_conn = []
start = 0
for size in cluster_sizes_conn.values:
    cluster_centers_conn.append(start + size / 2)
    start += size

ax.set_xticks(cluster_centers_conn)
ax.set_xticklabels([f"C{c}" for c in cluster_sizes_conn.index], rotation=0)

plt.title("Top 20 connection SHAP values ordered by existing clusters")
plt.xlabel("Subjects grouped by cluster")
plt.ylabel("Connections")
plt.tight_layout()
plt.savefig(
    os.path.join(cluster_plot_dir, "heatmap_top20_connections_ordered_by_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------------
# 7) PREPARE REGION SCORES PER SUBJECT
# ------------------------------------------------------------------
edge_shap_subject_mat = df_shap[shap_feature_cols].values  # [n_subjects x n_edges]
region_subject_scores = np.zeros((edge_shap_subject_mat.shape[0], 84), dtype=float)

for e_idx, (i, j) in enumerate(edge_pairs_1based):
    region_subject_scores[:, i - 1] += np.abs(edge_shap_subject_mat[:, e_idx])
    region_subject_scores[:, j - 1] += np.abs(edge_shap_subject_mat[:, e_idx])

region_subject_df = pd.DataFrame(
    region_subject_scores,
    columns=[clean_region_name(ROI_NAME_MAP[i]) for i in range(1, 85)]
)

region_subject_df["Subject_ID_fixed"] = df_shap["Subject_ID_fixed"].values

# top regions already computed above in region_importance_df
top_region_names = region_importance_df["Feature"].head(20).tolist()

region_plot_df = region_subject_df.merge(
    df_clusters[["Subject_ID_fixed", cluster_col]],
    on="Subject_ID_fixed",
    how="left"
)

region_plot_df = region_plot_df.dropna(subset=[cluster_col]).copy()
region_plot_df[cluster_col] = region_plot_df[cluster_col].astype(int)

region_plot_df = region_plot_df.sort_values([cluster_col, "Subject_ID_fixed"]).reset_index(drop=True)

region_heatmap_mat = region_plot_df[top_region_names].T

cluster_sizes_region = region_plot_df[cluster_col].value_counts().sort_index()
boundaries_region = np.cumsum(cluster_sizes_region.values)[:-1]

# ------------------------------------------------------------------
# 8) PLOT TOP REGIONS HEATMAP ORDERED BY CLUSTER
# ------------------------------------------------------------------
plt.figure(figsize=(16, 9))
ax = sns.heatmap(region_heatmap_mat, cmap="viridis")

for b in boundaries_region:
    ax.vlines(b, *ax.get_ylim(), colors="white", linewidth=2)

cluster_centers_region = []
start = 0
for size in cluster_sizes_region.values:
    cluster_centers_region.append(start + size / 2)
    start += size

ax.set_xticks(cluster_centers_region)
ax.set_xticklabels([f"C{c}" for c in cluster_sizes_region.index], rotation=0)

plt.title("Top 20 region scores ordered by existing clusters")
plt.xlabel("Subjects grouped by cluster")
plt.ylabel("Regions")
plt.tight_layout()
plt.savefig(
    os.path.join(cluster_plot_dir, "heatmap_top20_regions_ordered_by_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------------
# 9) SAVE SUBJECT ORDER USED IN THE HEATMAPS
# ------------------------------------------------------------------
conn_plot_df[["Subject_ID_fixed", cluster_col]].to_csv(
    os.path.join(cluster_plot_dir, "subject_order_connections_heatmap.csv"),
    index=False
)

region_plot_df[["Subject_ID_fixed", cluster_col]].to_csv(
    os.path.join(cluster_plot_dir, "subject_order_regions_heatmap.csv"),
    index=False
)

# ------------------------------------------------------------------
# 10) OPTIONAL: CLUSTER-WISE AVERAGE HEATMAPS
# ------------------------------------------------------------------
conn_cluster_mean_df = conn_plot_df.groupby(cluster_col)[top_conn_cols].mean().T
conn_cluster_mean_df.index = top_conn_names

plt.figure(figsize=(10, 8))
sns.heatmap(conn_cluster_mean_df, cmap="coolwarm", center=0)
plt.title("Mean top-connection SHAP per cluster")
plt.xlabel("Cluster")
plt.ylabel("Connections")
plt.tight_layout()
plt.savefig(
    os.path.join(cluster_plot_dir, "heatmap_mean_top_connections_per_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

region_cluster_mean_df = region_plot_df.groupby(cluster_col)[top_region_names].mean().T

plt.figure(figsize=(10, 8))
sns.heatmap(region_cluster_mean_df, cmap="viridis")
plt.title("Mean top-region score per cluster")
plt.xlabel("Cluster")
plt.ylabel("Regions")
plt.tight_layout()
plt.savefig(
    os.path.join(cluster_plot_dir, "heatmap_mean_top_regions_per_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------------
# 11) SAVE TABLES
# ------------------------------------------------------------------
conn_cluster_mean_df.to_csv(
    os.path.join(cluster_plot_dir, "mean_top_connections_per_cluster.csv")
)

region_cluster_mean_df.to_csv(
    os.path.join(cluster_plot_dir, "mean_top_regions_per_cluster.csv")
)

print("\n=== DONE: HEATMAPS ORDERED BY EXISTING CLUSTERS SAVED IN ===")
print(cluster_plot_dir)