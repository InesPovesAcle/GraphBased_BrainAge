#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDECODE healthy training with SHAP-guided contrastive learning
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
- Train final model on all healthy subjects
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

# IMPORTANT:
# This file must contain EDGE-LEVEL SHAP values, not compressed embeddings.
# Expected shape per row: Subject_ID + 3486 SHAP values for the upper-triangular edges.
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

#################### REMOVE AD / MCI ####################

if "Risk" in df_matched_connectomes.columns:
    risk_filled = df_matched_connectomes["Risk"].fillna("NoRisk").replace(r"^\s*$", "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()

print("FILTERING OUT AD AND MCI SUBJECTS")

df_matched_addecode_healthy = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
].copy()

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

print(f"Connectomes selected (excluding AD/MCI): {len(matched_connectomes_healthy_addecode)}")
print()

########################### PCA GENES ###############################

print("PCA GENES")

df_pca = pd.read_csv(pca_path)

df_pca["ID_fixed"] = df_pca["ID"].astype(str).str.upper().str.replace("_", "", regex=False)
df_matched_addecode_healthy["IDRNA_fixed"] = (
    df_matched_addecode_healthy["IDRNA"].astype(str).str.upper().str.replace("_", "", regex=False)
)

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_healthy_withConnectome = df_matched_addecode_healthy.merge(
    df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed"
)

print(f"Healthy subjects with metadata connectome: {df_matched_addecode_healthy.shape[0]}")
print(f"Healthy subjects with metadata PCA & connectome: {df_metadata_PCA_healthy_withConnectome.shape[0]}")
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

def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

####################### NODEWISE CLUSTERING #############################

def compute_nodewise_clustering_coefficients(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]
    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)

####################### THRESHOLD + LOG CONNECTOMES #############################

def threshold_connectome(matrix, percentile=100):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

log_thresholded_connectomes = {}
for subject, matrix in matched_connectomes_healthy_addecode.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=70)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)

####################### GRAPH METRICS #############################

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

for subject, matrix_log in log_thresholded_connectomes.items():
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
        print(f"Failed to compute metrics for subject {subject}: {e}")

####################### NORMALIZE GLOBAL FEATURES #############################

le_sex = LabelEncoder()
addecode_healthy_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_healthy_metadata_pca["sex"].astype(str))

le = LabelEncoder()
addecode_healthy_metadata_pca["genotype"] = le.fit_transform(addecode_healthy_metadata_pca["genotype"].astype(str))

numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ["PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"]

addecode_healthy_metadata_pca[numerical_cols] = addecode_healthy_metadata_pca[numerical_cols].apply(zscore)
addecode_healthy_metadata_pca[pca_cols] = addecode_healthy_metadata_pca[pca_cols].apply(zscore)

####################### BUILD GLOBAL FEATURE TENSORS #############################

subject_to_demographic_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_healthy_metadata_pca.iterrows()
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
    """
    Convert one connectome matrix into a PyG graph.
    """

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

################# CONVERT MATRIX TO GRAPH ################

graph_data_list_addecode = []
final_subjects_with_all_data = []

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_graphmetric_tensor:
            continue
        if subject not in subject_to_pca_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue
        if subject not in subject_to_shap_edge_scores:
            continue

        edge_index, edge_attr, node_features, shap_edge_scores = matrix_to_graph(
            matrix_log,
            subject_id=subject,
            node_features_dict=normalized_node_features_dict,
            shap_edge_dict=subject_to_shap_edge_scores
        )

        age_row = addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue

        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        demo_tensor = subject_to_demographic_tensor[subject]
        graph_tensor = subject_to_graphmetric_tensor[subject]
        pca_tensor = subject_to_pca_tensor[subject]
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
        final_subjects_with_all_data.append(subject)

        if len(graph_data_list_addecode) == 1:
            print("\nExample PyTorch Geometric Data object:")
            print("→ Node features shape:", data.x.shape)
            print("→ Edge index shape:", data.edge_index.shape)
            print("→ Edge attr shape:", data.edge_attr.shape)
            print("→ SHAP edge score shape:", data.shap_edge_scores.shape)
            print("→ Global features shape:", data.global_features.shape)
            print("→ Target age (y):", data.y.item())

    except Exception as e:
        print(f"Failed to process subject {subject}: {e}")

torch.save(graph_data_list_addecode, os.path.join(output_dir, "graph_data_list_addecode.pt"))
print("Saved:", os.path.join(output_dir, "graph_data_list_addecode.pt"))

print()
expected = set(subject_to_pca_tensor.keys())
actual = set(final_subjects_with_all_data)
missing = expected - actual

print(f"Subjects with PCA but no graph: {missing}")
print(f"Total graphs created: {len(actual)} / Expected: {len(expected)}")
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

print("Final matched lengths:")
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

    #############################
    # 1) CONTRASTIVE PRETRAINING ONCE PER OUTER FOLD
    #############################

    pretrained_encoder_state = pretrain_encoder_once_per_fold(
        train_graphs=outer_train_data,
        fold_id=fold + 1,
        batch_size=batch_size,
        contrastive_epochs=contrastive_epochs,
        contrastive_lr=contrastive_lr
    )

    #############################
    # 2) SUPERVISED REPEATS
    #############################

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

        #############################
        # 3) SUPERVISED AGE REGRESSION
        #    START FROM SAME PRETRAINED ENCODER
        #############################

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

        #############################
        # 4) LOAD BEST MODEL
        #############################

        best_encoder = GraphEncoder()
        best_model = BrainAgeRegressor(best_encoder).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()

        #############################
        # 5) GET TRAIN PREDICTIONS
        #############################

        y_train_true, y_train_pred_raw, _ = get_predictions(best_model, outer_train_loader_eval)
        a, b = fit_bias_correction(y_train_true, y_train_pred_raw)

        #############################
        # 6) GET TEST PREDICTIONS
        #############################

        y_test_true, y_test_pred_raw, ids_test = get_predictions(best_model, outer_test_loader)
        y_test_pred_bc = apply_bias_correction(y_test_pred_raw, a, b)

        #############################
        # 7) STORE GLOBAL PREDICTIONS
        #############################

        all_y_true_raw.extend(y_test_true.tolist())
        all_y_pred_raw.extend(y_test_pred_raw.tolist())
        all_y_pred_bc.extend(y_test_pred_bc.tolist())
        all_subject_ids.extend(ids_test)

        #############################
        # 8) METRICS FOR THIS REPEAT
        #############################

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

###################### SAVE PREDICTIONS #########################

df_preds = pd.DataFrame({
    "Subject_ID": all_subject_ids,
    "Real_Age": all_y_true_raw,
    "Predicted_Age_RAW": all_y_pred_raw,
    "Predicted_Age_BiasCorrected": all_y_pred_bc,
    "Brain_Age_Gap_RAW": all_y_pred_raw - all_y_true_raw,
    "Brain_Age_Gap_BiasCorrected": all_y_pred_bc - all_y_true_raw
})
df_preds.to_csv(os.path.join(output_dir, "cv_predictions_shap_guided_contrastive_learning.csv"), index=False)

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
plt.scatter(all_y_true_raw, all_y_pred_raw, alpha=0.7, edgecolors="k", label="Predictions RAW")

min_val = min(min(all_y_true_raw), min(all_y_pred_raw))
max_val = max(max(all_y_true_raw), max(all_y_pred_raw))
margin = (max_val - min_val) * 0.05

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")

reg_raw = LinearRegression().fit(all_y_true_raw.reshape(-1, 1), all_y_pred_raw)
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
    f"MAE: {all_maes_raw.mean():.2f} ± {all_maes_raw.std():.2f}\n"
    f"RMSE: {all_rmses_raw.mean():.2f} ± {all_rmses_raw.std():.2f}\n"
    f"R²: {all_r2s_raw.mean():.2f} ± {all_r2s_raw.std():.2f}\n"
    f"BAG slope: {global_raw_bag_slope:.4f}"
)

plt.text(
    0.95,
    0.05,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=11,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
)

plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Ages (RAW)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SHAPguidedCL_scatter_raw.png"), dpi=300)
plt.close()

###################### SCATTER BIAS-CORRECTED #########################

plt.figure(figsize=(8, 6))
plt.scatter(all_y_true_raw, all_y_pred_bc, alpha=0.7, edgecolors="k", label="Predictions Bias-Corrected")

min_val = min(min(all_y_true_raw), min(all_y_pred_bc))
max_val = max(max(all_y_true_raw), max(all_y_pred_bc))
margin = (max_val - min_val) * 0.05

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")

reg_bc = LinearRegression().fit(all_y_true_raw.reshape(-1, 1), all_y_pred_bc)
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
    f"MAE: {all_maes_bc.mean():.2f} ± {all_maes_bc.std():.2f}\n"
    f"RMSE: {all_rmses_bc.mean():.2f} ± {all_rmses_bc.std():.2f}\n"
    f"R²: {all_r2s_bc.mean():.2f} ± {all_r2s_bc.std():.2f}\n"
    f"BAG slope: {global_bc_bag_slope:.4f}"
)

plt.text(
    0.95,
    0.05,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=11,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
)

plt.xlabel("Real Age")
plt.ylabel("Predicted Age (Bias-Corrected)")
plt.title("Predicted vs Real Ages (Bias-Corrected)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SHAPguidedCL_scatter_bias_corrected.png"), dpi=300)
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

torch.save(final_model.state_dict(), os.path.join(output_dir, "final_model_trained_on_all_healthy.pt"))
print("\nFinal model saved as:", os.path.join(output_dir, "final_model_trained_on_all_healthy.pt"))