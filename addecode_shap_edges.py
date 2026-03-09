#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AD-DECODE
- Preprocess risk data using healthy-only normalization
- Build multimodal graphs
- Load pretrained BrainAge GATv2 model
- Run EDGE-SHAP on connectome edges
"""

#################  IMPORT NECESSARY LIBRARIES  ################

import os
import re
import zipfile
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

import shap

warnings.filterwarnings("ignore")

#################  PATHS  ################

WORK = os.environ["WORK"]

RESULTS_DIR = os.path.join(WORK, "ines/results/Shap_edges")
os.makedirs(RESULTS_DIR, exist_ok=True)

EDGE_SHAP_DIR = os.path.join(RESULTS_DIR, "edges_addecode")
os.makedirs(EDGE_SHAP_DIR, exist_ok=True)

print(f"Results dir: {RESULTS_DIR}")
print(f"Edge SHAP dir: {EDGE_SHAP_DIR}")

#################  REPRODUCIBILITY  ################

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

####################### CONNECTOMES ###############################

print("\nADDECODE CONNECTOMES\n")

zip_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/connectomes/AD_DECODE_connectome_act.zip"
)

directory_inside_zip = "connectome_act/"
connectomes = {}

with zipfile.ZipFile(zip_path, "r") as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                df = pd.read_csv(f, header=None)
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[subject_id] = df

print(f"Total connectome matrices loaded: {len(connectomes)}")

filtered_connectomes = {
    k: v for k, v in connectomes.items() if "_whitematter" not in k
}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)
    if match:
        num_id = match.group(1).zfill(5)
        cleaned_connectomes[num_id] = v

print(f"Cleaned connectomes: {len(cleaned_connectomes)}")

############################## METADATA ##############################

print("\nADDECODE METADATA\n")

metadata_path = os.path.join(WORK, "ines/data/AD_DECODE_data4.xlsx")
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

#################### MATCH CONNECTOMES & METADATA ####################

print("\nMATCHING CONNECTOMES WITH METADATA\n")

matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(
    f"Matched subjects (metadata & connectome): "
    f"{len(matched_metadata)} out of {len(cleaned_connectomes)}"
)

matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}

df_matched_connectomes = matched_metadata.copy()

########################### PCA GENES ###############################

print("\nPCA GENES\n")

pca_path = os.path.join(WORK, "ines/data/PCA_human_blood_top30.csv")
df_pca = pd.read_csv(pca_path)

df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)
df_matched_connectomes["IDRNA_fixed"] = (
    df_matched_connectomes["IDRNA"].astype(str).str.upper().str.replace("_", "", regex=False)
)

print("\nMATCH PCA GENES WITH METADATA\n")

df_metadata_PCA_withConnectome = df_matched_connectomes.merge(
    df_pca,
    how="inner",
    left_on="IDRNA_fixed",
    right_on="ID_fixed"
)

print(f"Subjects with metadata + connectome: {df_matched_connectomes.shape[0]}")
print(f"Subjects with metadata + PCA + connectome: {df_metadata_PCA_withConnectome.shape[0]}")

all_ids = set(df_matched_connectomes["MRI_Exam_fixed"])
with_pca_ids = set(df_metadata_PCA_withConnectome["MRI_Exam_fixed"])
without_pca_ids = all_ids - with_pca_ids

df_without_pca = df_matched_connectomes[
    df_matched_connectomes["MRI_Exam_fixed"].isin(without_pca_ids)
]

print(f"Subjects with connectome but NO PCA: {df_without_pca.shape[0]}")
if not df_without_pca.empty:
    print(df_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]].head())

####################### FA MD VOL #############################

print("\nFA / MD / VOLUME\n")

fa_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt",
)
md_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt",
)
vol_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume_norm.txt",
)

# FA
df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"]
df_fa = df_fa.reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col.startswith("S")]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)

# MD
df_md = pd.read_csv(md_path, sep="\t")
df_md = df_md[1:]
df_md = df_md[df_md["ROI"] != "0"]
df_md = df_md.reset_index(drop=True)
subject_cols_md = [col for col in df_md.columns if col.startswith("S")]
df_md_transposed = df_md[subject_cols_md].transpose()
df_md_transposed.columns = [f"ROI_{i+1}" for i in range(df_md_transposed.shape[1])]
df_md_transposed.index.name = "subject_id"
df_md_transposed = df_md_transposed.astype(float)

# VOL
df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"]
df_vol = df_vol.reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col.startswith("S")]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

multimodal_features_dict = {}

for subj in df_fa_transposed.index:
    subj_id = subj.replace("S", "").zfill(5)
    if subj in df_md_transposed.index and subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float)
        stacked = torch.stack([fa, md, vol], dim=1)  # [84, 3]
        multimodal_features_dict[subj_id] = stacked

print(f"Subjects with multimodal node features: {len(multimodal_features_dict)}")

####################### HEALTHY-ONLY NODE NORMALIZATION #############################

healthy_ids = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
]["MRI_Exam_fixed"].tolist()

healthy_stack = torch.stack([
    multimodal_features_dict[subj]
    for subj in healthy_ids
    if subj in multimodal_features_dict
])

node_means = healthy_stack.mean(dim=0)  # [84, 3]
node_stds = healthy_stack.std(dim=0) + 1e-8

normalized_node_features_dict = {
    subj: (features - node_means) / node_stds
    for subj, features in multimodal_features_dict.items()
}

print(f"Healthy controls used for node normalization: {healthy_stack.shape[0]}")

####################### NODEWISE CLUSTERING #############################

def compute_nodewise_clustering_coefficients(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())

    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]

    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]

    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)  # [84,1]

####################### THRESHOLD + LOG CONNECTOMES #############################

def threshold_connectome(matrix, percentile=95):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

log_thresholded_connectomes = {}
for subject, matrix in matched_connectomes.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes[subject] = pd.DataFrame(
        log_matrix, index=matrix.index, columns=matrix.columns
    )

print(f"Thresholded/log-transformed connectomes: {len(log_thresholded_connectomes)}")

####################### MATRIX TO GRAPH #############################

def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)

    edge_index = torch.tensor(
        np.vstack(indices),
        dtype=torch.long,
        device=device
    )

    edge_attr = torch.tensor(
        matrix.values[indices],
        dtype=torch.float,
        device=device
    ).unsqueeze(-1)  # [E,1] important for edge_dim=1

    node_feats = node_features_dict[subject_id]  # [84,3]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)  # [84,1]

    full_node_features = torch.cat([node_feats, clustering_tensor], dim=1)  # [84,4]
    node_features = 0.5 * full_node_features.to(device)

    return edge_index, edge_attr, node_features

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

addecode_metadata_pca = df_metadata_PCA_withConnectome.reset_index(drop=True)
addecode_metadata_pca["Clustering_Coeff"] = np.nan
addecode_metadata_pca["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)

        addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "Clustering_Coeff"
        ] = clustering

        addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "Path_Length"
        ] = path

    except Exception as e:
        print(f"Failed to compute metrics for subject {subject}: {e}")

####################### GLOBAL FEATURE NORMALIZATION #############################

le_sex = LabelEncoder()
addecode_metadata_pca["sex_encoded"] = le_sex.fit_transform(
    addecode_metadata_pca["sex"].astype(str)
)

le_gen = LabelEncoder()
addecode_metadata_pca["genotype"] = le_gen.fit_transform(
    addecode_metadata_pca["genotype"].astype(str)
)

numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ["PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"]

df_controls = addecode_metadata_pca[
    ~addecode_metadata_pca["Risk"].isin(["AD", "MCI"])
]

all_zscore_cols = numerical_cols + pca_cols

global_means = df_controls[all_zscore_cols].mean()
global_stds = df_controls[all_zscore_cols].std() + 1e-8

addecode_metadata_pca[all_zscore_cols] = (
    addecode_metadata_pca[all_zscore_cols] - global_means
) / global_stds

####################### BUILD GLOBAL TENSORS #############################

subject_to_demographic_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_metadata_pca.iterrows()
}

subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_metadata_pca.iterrows()
}

subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(
        row[pca_cols].values.astype(np.float32)
    )
    for _, row in addecode_metadata_pca.iterrows()
}

####################### CONVERT ALL SUBJECTS TO GRAPHS #############################

print("\nCONVERT MATRIX TO GRAPH\n")

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

        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log,
            device=torch.device("cpu"),
            subject_id=subject,
            node_features_dict=normalized_node_features_dict
        )

        age_row = addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue

        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        demo_tensor = subject_to_demographic_tensor[subject]   # [4]
        graph_tensor = subject_to_graphmetric_tensor[subject]  # [2]
        pca_tensor = subject_to_pca_tensor[subject]            # [10]

        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)  # [16]

        data = Data(
            x=node_features,                         # [84,4]
            edge_index=edge_index,                  # [2,E]
            edge_attr=edge_attr,                    # [E,1]
            y=age,                                  # [1]
            global_features=global_feat.unsqueeze(0)  # [1,16]
        )
        data.subject_id = subject

        graph_data_list_addecode.append(data)
        final_subjects_with_all_data.append(subject)

        if len(graph_data_list_addecode) == 1:
            print("Example PyTorch Geometric Data object:")
            print("  Node features shape:", data.x.shape)
            print("  Edge index shape:", data.edge_index.shape)
            print("  Edge attr shape:", data.edge_attr.shape)
            print("  Global features shape:", data.global_features.shape)
            print("  Target age (y):", data.y.item())

    except Exception as e:
        print(f"Failed to process subject {subject}: {e}")

print(f"\nTotal graphs created: {len(graph_data_list_addecode)}")

if len(graph_data_list_addecode) == 0:
    raise RuntimeError(
        "graph_data_list_addecode is empty. No graphs were created, so SHAP cannot run."
    )

####################### MODEL #############################

print("\nLOAD PRETRAINED MODEL\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BrainAgeGATv2(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Uses edge_attr
        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

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
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.node_embed(x)

        x = self.gnn1(x, edge_index, edge_attr=data.edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index, edge_attr=data.edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr=data.edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr=data.edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)

        global_feats = data.global_features.to(x.device).squeeze(1)

        meta_embed = self.meta_head(global_feats[:, 0:4])
        graph_embed = self.graph_head(global_feats[:, 4:6])
        pca_embed = self.pca_head(global_feats[:, 6:])

        global_embed = torch.cat([meta_embed, graph_embed, pca_embed], dim=1)
        x = torch.cat([x, global_embed], dim=1)

        x = self.fc(x)
        return x

model = BrainAgeGATv2(global_feat_dim=16).to(device)

model_path = os.path.join(WORK, "ines/code/model_trained_on_all_healthy.pt")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model from: {model_path}")

####################### EDGE SHAP WRAPPER #############################

class EdgeSHAPWrapperAD(torch.nn.Module):
    """
    Only varies edge_attr.
    Keeps fixed:
      - x
      - edge_index
      - global_features
    """

    def __init__(self, model: torch.nn.Module, base_data: Data):
        super().__init__()
        self.model = model
        self.base_data = base_data.clone().to(next(model.parameters()).device)

        if not hasattr(self.base_data, "batch") or self.base_data.batch is None:
            self.base_data.batch = torch.zeros(
                self.base_data.num_nodes,
                dtype=torch.long,
                device=self.base_data.x.device
            )

    def forward(self, edge_attr_batch):
        outputs = []

        for ea in edge_attr_batch:
            d = self.base_data.clone()

            # Ensure shape [E,1]
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            elif ea.dim() == 2 and ea.shape[-1] != 1:
                ea = ea.unsqueeze(-1)

            d.edge_attr = ea.to(d.x.device)

            out = self.model(d)   # shape [1,1] normalmente
            out = out.view(1, 1)  # fuerza shape [1,1]
            outputs.append(out)

        # Final shape: [B,1]
        return torch.cat(outputs, dim=0)

####################### RUN EDGE SHAP #############################

print("\nRUNNING EDGE SHAP\n")

all_subject_summaries = []

for idx, data in enumerate(graph_data_list_addecode, 1):
    sid = data.subject_id
    print(f"[{idx}/{len(graph_data_list_addecode)}] Subject {sid}")

    try:
        base_data = data.clone().to(device)

        if base_data.edge_attr.dim() == 1:
            base_data.edge_attr = base_data.edge_attr.unsqueeze(-1)

        wrapper = EdgeSHAPWrapperAD(model, base_data)

        num_edges = base_data.edge_attr.shape[0]

        # Keep edge attributes in shape [1, E, 1]
        baseline = torch.zeros((1, num_edges, 1), dtype=torch.float32, device=device)
        input_ea = base_data.edge_attr.unsqueeze(0)  # [1, E, 1]
        
        explainer = shap.GradientExplainer(wrapper, baseline)
        shap_vals = explainer.shap_values(input_ea)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        shap_vals = np.array(shap_vals)
        
        # Expected shape could be [1, E, 1] or [1, E]
        if shap_vals.ndim == 3:
            shap_edge = shap_vals[0, :, 0]
        elif shap_vals.ndim == 2:
            shap_edge = shap_vals[0, :]
        else:
            shap_edge = np.squeeze(shap_vals)
        edges = base_data.edge_index.detach().cpu().numpy().T  # [E,2]

        with torch.no_grad():
            pred_age = float(model(base_data).detach().cpu().item())

        df_out = pd.DataFrame({
            "Node_i": edges[:, 0],
            "Node_j": edges[:, 1],
            "edge_weight": base_data.edge_attr.detach().cpu().numpy().squeeze(),
            "SHAP_val": shap_edge,
            "abs_SHAP": np.abs(shap_edge)
        })

        out_csv = os.path.join(EDGE_SHAP_DIR, f"edge_shap_subject_{sid}.csv")
        df_out.to_csv(out_csv, index=False)

        all_subject_summaries.append({
            "Subject_ID": sid,
            "Pred_Age": pred_age,
            "MeanAbsSHAP": float(np.mean(np.abs(shap_edge))),
            "MaxAbsSHAP": float(np.max(np.abs(shap_edge))),
        })

        print(f"    Saved: {out_csv}")

    except Exception as e:
        print(f"    Failed for subject {sid}: {e}")

####################### SAVE GLOBAL SUMMARY #############################

if len(all_subject_summaries) > 0:
    df_summary = pd.DataFrame(all_subject_summaries)
    summary_csv = os.path.join(EDGE_SHAP_DIR, "edge_shap_summary_all_subjects.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

####################### AVERAGE EDGE IMPORTANCE #############################

all_edge_tables = []

for fname in os.listdir(EDGE_SHAP_DIR):
    if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
        fpath = os.path.join(EDGE_SHAP_DIR, fname)
        df_tmp = pd.read_csv(fpath)
        df_tmp["edge_key"] = (
            df_tmp["Node_i"].astype(str) + "_" + df_tmp["Node_j"].astype(str)
        )
        all_edge_tables.append(df_tmp[["edge_key", "Node_i", "Node_j", "abs_SHAP"]])

if len(all_edge_tables) > 0:
    df_all_edges = pd.concat(all_edge_tables, ignore_index=True)

    df_mean_edges = (
        df_all_edges
        .groupby(["edge_key", "Node_i", "Node_j"], as_index=False)["abs_SHAP"]
        .mean()
        .rename(columns={"abs_SHAP": "mean_abs_SHAP"})
        .sort_values("mean_abs_SHAP", ascending=False)
    )

    mean_csv = os.path.join(EDGE_SHAP_DIR, "edge_shap_mean_abs_all_subjects.csv")
    df_mean_edges.to_csv(mean_csv, index=False)
    print(f"Saved mean edge importance: {mean_csv}")

    top_n = 20
    df_top = df_mean_edges.head(top_n).copy()
    df_top["Edge"] = df_top["Node_i"].astype(str) + "-" + df_top["Node_j"].astype(str)
    df_top = df_top.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(df_top["Edge"], df_top["mean_abs_SHAP"])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top {top_n} most important edges across subjects")
    plt.tight_layout()

    top_fig = os.path.join(EDGE_SHAP_DIR, "edge_shap_top20_mean_abs.png")
    plt.savefig(top_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {top_fig}")

print("\nDONE.")