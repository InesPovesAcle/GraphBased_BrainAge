#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:18:33 2026

@author: ines
"""

#  Fine-Tune All Layers (Full Fine-Tuning) 

#Same features and node features in adni and addecode 


#unfreeze all layers of the pre-trained model, 
#allow their weights to be updated during training on the target dataset. 
#This approach enables the model to adapt more to the new data. 



# ==== BLOCK 1: SETUP AND IMPORTS ====

# Standard libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import re

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# NetworkX for graph metrics
import networkx as nx

# ==== Set seed for reproducibility ====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
work_path='/mnt/newStor/paros/paros_WORK/'
# ==== Check if CUDA is available ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

adni_model_path = os.path.join(
    os.environ["WORK"],
    "ines/results/BrainAgePredictionADNI/brainage_adni1_prediction_model_RAW.pt"
)

print("Loading pretrained ADNI model from:")
print(adni_model_path)

print("Loading pretrained ADNI model from:", adni_model_path)
if not os.path.exists(adni_model_path):
    raise FileNotFoundError(adni_model_path)

adni_ckpt = torch.load(adni_model_path, map_location=device)
adni_state_dict = adni_ckpt["model_state_dict"]  # <-- esto es lo que quieres para transfer learning

# ==== BLOCK 2: GATv2 MODEL FOR AD-DECODE ====

OUT_DIR = os.path.join(
    os.environ["WORK"],
    "ines/results/addecode_training_with_adni_model"
)

CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Results directory:", OUT_DIR)
print("Checkpoint directory:", CHECKPOINT_DIR)

######################  DEFINE MODEL  #########################

# MULTIHEAD-> one head for each global feature

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class BrainAgeGATv2(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()

        # === NODE FEATURES EMBEDDING ===
        # Each brain region (node) has 4 features: FA, MD, Volume, Clustering coefficient.
        # These are embedded into a higher-dimensional representation (64).
        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),  # Project node features to 64-dimensional space
            nn.ReLU(),
            nn.Dropout(0.15)
            
        )

        # === GATv2 LAYERS WITH EDGE ATTRIBUTES ===
        # These layers use the connectome (edge weights) to propagate information.
        # edge_dim=1 means each edge has a scalar weight (from the functional connectome).
        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)  # Normalize output (16*8 = 128 channels)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)  # Regularization

        # === GLOBAL FEATURE BRANCHES ===
        # These process metadata that is not node-specific, grouped into 3 categories.

        # Demographic + physiological metadata (sex, systolic, diastolic, genotype)
        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Graph-level metrics: global clustering coefficient and path length
        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Top 10 PCA components from gene expression data, selected for age correlation
        self.pca_head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # === FINAL FUSION MLP ===
        # Combines graph-level information from GNN and global features
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 128),  # 128 from GNN output + 64 from metadata branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Final output: predicted brain age
        )

    def forward(self, data):
        # === GRAPH INPUTS ===
        x = data.x               # Node features: shape [num_nodes, 4]
        edge_index = data.edge_index  # Graph connectivity (edges)
        edge_attr = data.edge_attr    # Edge weights from functional connectome

        # === NODE EMBEDDING ===
        x = self.node_embed(x)  # Embed the node features

        # === GNN BLOCK 1 ===
        x = self.gnn1(x, edge_index, edge_attr=edge_attr)  # Attention using connectome weights
        x = self.bn1(x)
        x = F.relu(x)

        # === GNN BLOCK 2 with residual connection ===
        x_res1 = x  # Save for residual
        x = self.gnn2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        # === GNN BLOCK 3 with residual ===
        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        # === GNN BLOCK 4 with residual ===
        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        # === POOLING ===
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)  # Aggregate node embeddings into graph-level representation

        # === GLOBAL FEATURES ===
        # Shape: [batch_size, 1, 16] → remove extra dimension
        global_feats = data.global_features.to(x.device).squeeze(1)

        # Process each global feature group
        meta_embed = self.meta_head(global_feats[:, 0:4])    # Demographics
        graph_embed = self.graph_head(global_feats[:, 4:6])  # Clustering and path length
        pca_embed = self.pca_head(global_feats[:, 6:])       # Top 10 gene PCs

        # Concatenate all global embeddings
        global_embed = torch.cat([meta_embed, graph_embed, pca_embed], dim=1)  # Shape: [batch_size, 64]

        # === FUSION AND PREDICTION ===
        x = torch.cat([x, global_embed], dim=1)  # Combine GNN and metadata features
        x = self.fc(x)  # Final MLP to predict age

        return x  # Output: predicted age




# ==== BLOCK 3: LOAD PRETRAINED WEIGHTS (TRANSFER LEARNING) ====

# 1. Instantiate the model (same architecture as in ADNI)
model = BrainAgeGATv2(global_feat_dim=16).to(device)

# 2. Load pretrained weights from ADNI (make sure the path is correct)

ckpt = torch.load(adni_model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded ADNI pretrained weights.")

# 3. Load all weights directly (no exclusions needed)


# 4. Unfreeze all layers so they are trainable during fine-tuning
for param in model.parameters():
    param.requires_grad = True





#4
# ADDECODE Data



####################### CONNECTOMES ###############################
print("ADDECODE CONNECTOMES\n")
work_path='/mnt/newStor/paros/paros_WORK/'
# === Define paths ===
#zip_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_connectome_act.zip"
#zip_path = '/$WORK/ines/data/harmonization/ADDecode/connectomes/AD_DECODE_connectome_act.zip'
#zip_path = '/mnt/newStor/paros/paros_WORK//ines/data/harmonization/ADDecode/connectomes/AD_DECODE_connectome_act.zip'



zip_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/connectomes/AD_DECODE_connectome_act.zip"
)

directory_inside_zip = "connectome_act/"
connectomes = {}





# === Load connectome matrices from ZIP ===
with zipfile.ZipFile(zip_path, 'r') as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                df = pd.read_csv(f, header=None)
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[subject_id] = df

print(f"Total connectome matrices loaded: {len(connectomes)}")

# === Filter out connectomes with white matter on their file name ===
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# === Extract subject IDs from filenames ===
cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)
    if match:
        num_id = match.group(1).zfill(5)  # Ensure 5-digit IDs
        cleaned_connectomes[num_id] = v

print()


############################## METADATA ##############################


print("ADDECODE METADATA\n")

# === Load metadata CSV ===


metadata_path = os.path.join(
    os.environ["WORK"],
    "ines/data/AD_DECODE_data4.xlsx"
)
df_metadata = pd.read_excel(metadata_path)

# === Generate standardized subject IDs → 'DWI_fixed' (e.g., 123 → '00123')
df_metadata["MRI_Exam_fixed"] = (
    df_metadata["MRI_Exam"]
    .fillna(0)                           # Handle NaNs first
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# === Drop fully empty rows and those with missing DWI ===
df_metadata_cleaned = df_metadata.dropna(how='all')                       # Remove fully empty rows
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["MRI_Exam"])         # Remove rows without DWI

# === Display result ===
print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()





#################### MATCH CONNECTOMES & METADATA ####################

print(" MATCHING CONNECTOMES WITH METADATA")

# === Filter metadata to only subjects with connectomes available ===
matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

# === Build dictionary of matched connectomes ===
matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}


# === Store matched metadata as a DataFrame for further processing ===
df_matched_connectomes = matched_metadata.copy()





#Remove AD and MCI

# === Print risk distribution if available ===
if "Risk" in df_matched_connectomes.columns:
    risk_filled = df_matched_connectomes["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()



print("FILTERING OUT AD AND MCI SUBJECTS")

# === Keep only healthy control subjects ===
df_matched_addecode_healthy = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
].copy()

print(f"Subjects before removing AD/MCI: {len(df_matched_connectomes)}")
print(f"Subjects after removing AD/MCI: {len(df_matched_addecode_healthy)}")
print()


# === Show updated 'Risk' distribution ===
if "Risk" in df_matched_addecode_healthy.columns:
    risk_filled = df_matched_addecode_healthy["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()



#Connectomes
# === Filter connectomes to include only those from non-AD/MCI subjects ===
matched_connectomes_healthy_addecode = {
    row["MRI_Exam_fixed"]: matched_connectomes[row["MRI_Exam_fixed"]]
    for _, row in df_matched_addecode_healthy.iterrows()
}

# === Confirmation of subject count
print(f"Connectomes selected (excluding AD/MCI): {len(matched_connectomes_healthy_addecode)}")
print()


# df_matched_connectomes:
# → Cleaned metadata that has a valid connectome
# → Includes AD/MCI

# matched_connectomes:
# → Dictionary of connectomes that have valid metadata
# → Key: subject ID
# → Value: connectome matrix
# → Includes AD/MCI




# df_matched_addecode_healthy:
# → Metadata of only healthy subjects (no AD/MCI)
# → Subset of df_matched_connectomes

# matched_connectomes_healthy_addecode:
# → Connectomes of only healthy subjects
# → Subset of matched_connectomes





########### PCA GENES ##########

print("PCA GENES")

import pandas as pd

# Read 
df_pca = pd.read_csv("/mnt/newStor/paros/paros_WORK/ines/data/PCA_human_blood_top30.csv")
print(df_pca.head())

print(df_matched_addecode_healthy.head())



# Fix id formats

# === Fix ID format in PCA DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE_1' → 'ADDECODE1'
df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)



# === Fix Subject format in metadata DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE1' → 'ADDECODE1'
df_matched_addecode_healthy["IDRNA_fixed"] = df_matched_addecode_healthy["IDRNA"].str.upper().str.replace("_", "", regex=False)





###### MATCH PCA GENES WITH METADATA############

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_healthy_withConnectome = df_matched_addecode_healthy.merge(df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed")


#Numbers

# === Show how many healthy subjects with PCA and connectome you have
print(f" Healthy subjects with metadata connectome: {df_matched_addecode_healthy.shape[0]}")
print()

print(f" Healthy subjects with metadata PCA & connectome: {df_metadata_PCA_healthy_withConnectome.shape[0]}")
print()


# Get the full set of subject IDs (DWI_fixed) in healthy set
all_healthy_ids = set(df_matched_addecode_healthy["MRI_Exam_fixed"])

# Get the subject IDs (DWI_fixed) that matched with PCA
healthy_with_pca_ids = set(df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"])

# Compute the difference: healthy subjects without PCA
healthy_without_pca_ids = all_healthy_ids - healthy_with_pca_ids

# Filter the original healthy metadata for those subjects
df_healthy_without_pca = df_matched_addecode_healthy[
    df_matched_addecode_healthy["MRI_Exam_fixed"].isin(healthy_without_pca_ids)
]


# Print result
print(f" Healthy subjects with connectome but NO PCA: {df_healthy_without_pca.shape[0]}")
print()

print(df_healthy_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]])






####################### FA MD Vol #############################



# === Load FA data ===
fa_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt"
)

df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"]
df_fa = df_fa.reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col.startswith("S")]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)


import re

# Clean and deduplicate FA subjects based on numeric ID (e.g. "02842")
cleaned_fa = {}

for subj in df_fa_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_fa:
            cleaned_fa[subj_id] = df_fa_transposed.loc[subj]

# Convert cleaned data to DataFrame
df_fa_transposed_cleaned = pd.DataFrame.from_dict(cleaned_fa, orient="index")
df_fa_transposed_cleaned.index.name = "subject_id"



# === Load MD data ===
md_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt"
)
df_md = pd.read_csv(md_path, sep="\t")
df_md = df_md[1:]
df_md = df_md[df_md["ROI"] != "0"]
df_md = df_md.reset_index(drop=True)
subject_cols_md = [col for col in df_md.columns if col.startswith("S")]
df_md_transposed = df_md[subject_cols_md].transpose()
df_md_transposed.columns = [f"ROI_{i+1}" for i in range(df_md_transposed.shape[1])]
df_md_transposed.index.name = "subject_id"
df_md_transposed = df_md_transposed.astype(float)


# Clean and deduplicate MD subjects based on numeric ID (e.g. "02842")
cleaned_md = {}

for subj in df_md_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_md:
            cleaned_md[subj_id] = df_md_transposed.loc[subj]

df_md_transposed_cleaned = pd.DataFrame.from_dict(cleaned_md, orient="index")
df_md_transposed_cleaned.index.name = "subject_id"




# === Load Volume data ===
vol_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume.txt"
)
df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"]
df_vol = df_vol.reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col.startswith("S")]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)


# Clean and deduplicate Volume subjects based on numeric ID (e.g. "02842")
cleaned_vol = {}

for subj in df_vol_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_vol:
            cleaned_vol[subj_id] = df_vol_transposed.loc[subj]

df_vol_transposed_cleaned = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_transposed_cleaned.index.name = "subject_id"


# === Combine FA + MD + Vol per subject using cleaned DataFrames ===

multimodal_features_dict = {}

# Use subject IDs from FA as reference (already cleaned to 5-digit keys)
for subj_id in df_fa_transposed_cleaned.index:
    # Check that this subject also exists in MD and Vol
    if subj_id in df_md_transposed_cleaned.index and subj_id in df_vol_transposed_cleaned.index:
        fa = torch.tensor(df_fa_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed_cleaned.loc[subj_id].values, dtype=torch.float)

        # Stack the 3 modalities: [84 nodes, 3 features (FA, MD, Vol)]
        stacked = torch.stack([fa, md, vol], dim=1)

        # Store with subject ID as key
        multimodal_features_dict[subj_id] = stacked



print()
print(" Subjects with FA, MD, and Vol features:", len(multimodal_features_dict))

fa_md_vol_ids = set(multimodal_features_dict.keys())
pca_ids = set(df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"])
connectome_ids = set(matched_connectomes_healthy_addecode.keys())

final_overlap = fa_md_vol_ids & pca_ids & connectome_ids

print(" Subjects with FA/MD/Vol + PCA + Connectome:", len(final_overlap))

# Sample one subject from the dictionary
example_id = list(multimodal_features_dict.keys())[25]
print(" Example subject ID:", example_id)

# Check that this subject also exists in metadata and connectomes
in_metadata = example_id in df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"].values
in_connectome = example_id in matched_connectomes_healthy_addecode

print(f" In metadata: {in_metadata}")
print(f" In connectomes: {in_connectome}")

# Print first few FA/MD/Vol values before normalization
example_tensor = multimodal_features_dict[example_id]
print(" First 5 nodes (FA):", example_tensor[:5, 0])
print(" First 5 nodes (MD):", example_tensor[:5, 1])
print(" First 5 nodes (Vol):", example_tensor[:5, 2])
print()





# === Normalization node-wise  ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # [N_subjects, 84, 3]
    means = all_features.mean(dim=0)  # [84, 3]
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Normalization
normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)





# === Function to compute clustering coefficient per node ===
def compute_nodewise_clustering_coefficients(matrix):
    """
    Compute clustering coefficient for each node in the connectome matrix.
    
    Parameters:
        matrix (pd.DataFrame): 84x84 connectivity matrix
    
    Returns:
        torch.Tensor: Tensor of shape [84, 1] with clustering coefficient per node
    """
    G = nx.from_numpy_array(matrix.to_numpy())

    # Assign weights from matrix to the graph
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]

    # Compute clustering coefficient per node
    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]

    # Convert to tensor [84, 1]
    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)








# ===============================
# Step 9: Threshold and Log Transform Connectomes
# ===============================

import numpy as np
import pandas as pd

# --- Define thresholding function ---
def threshold_connectome(matrix, keep_top_percent=70):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    thr = np.percentile(values, 100 - keep_top_percent)  # top keep_top_percent%
    thresholded_np = np.where(matrix_np >= thr, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

# --- Apply threshold + log transform ---
log_thresholded_connectomes = {}
for subject, matrix in matched_connectomes_healthy_addecode.items():
    thresholded_matrix = threshold_connectome(matrix, keep_top_percent=70)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)



##################### MATRIX TO GRAPH FUNCTION #######################

import torch
import numpy as np
from torch_geometric.data import Data


# === Function to convert a connectome matrix into a graph with multimodal node features ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)

    # === Get FA, MD, Volume features [84, 3]
    node_feats = node_features_dict[subject_id]

    # === Compute clustering coefficient per node [84, 1]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)

    # === Concatenate and scale [84, 4]
    full_node_features = torch.cat([node_feats, clustering_tensor], dim=1)
    node_features = 0.5 * full_node_features.to(device)

    return edge_index, edge_attr, node_features






# ===============================
# Step 10: Compute Graph Metrics and Add to Metadata
# ===============================

import networkx as nx

# --- Define graph metric functions ---
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
    except:
        return float("nan")

# --- Assign computed metrics to metadata ---
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


# ===============================
# Step 11: Normalize Metadata and PCA Columns
# ===============================

from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

#label encoding sex
le_sex = LabelEncoder()
addecode_healthy_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_healthy_metadata_pca["sex"].astype(str))


# --- Label encode genotype ---
le = LabelEncoder()
addecode_healthy_metadata_pca["genotype"] = le.fit_transform(addecode_healthy_metadata_pca["genotype"].astype(str))

# --- Normalize numerical and PCA columns ---
numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3'] #Top 10 from SPEARMAN  corr (enrich)

addecode_healthy_metadata_pca[numerical_cols] = addecode_healthy_metadata_pca[numerical_cols].apply(zscore)
addecode_healthy_metadata_pca[pca_cols] = addecode_healthy_metadata_pca[pca_cols].apply(zscore)



# ===============================
# Step 12: Build Metadata, graph metrics and PCA Tensors
# ===============================

# === 1. Demographic tensor (systolic, diastolic, sex one-hot, genotype) ===
subject_to_demographic_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

# === 2. Graph metric tensor (clustering coefficient, path length) ===
subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

# === 3. PCA tensor (top 10 age-correlated components) ===
subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_healthy_metadata_pca.iterrows()
}




#################  CONVERT MATRIX TO GRAPH  ################

graph_data_list_addecode = []
final_subjects_with_all_data = []  # Para verificar qué sujetos sí se procesan

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        # === Skip if any required input is missing ===
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_graphmetric_tensor:
            continue
        if subject not in subject_to_pca_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # === Convert matrix to graph (node features: FA, MD, Vol, clustering)
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device=torch.device("cpu"), subject_id=subject, node_features_dict=normalized_node_features_dict
        )

        # === Get target age
        age_row = addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue
        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        # === Concatenate global features (demographics + graph metrics + PCA)
        demo_tensor = subject_to_demographic_tensor[subject]     # [5]
        graph_tensor = subject_to_graphmetric_tensor[subject]    # [2]
        pca_tensor = subject_to_pca_tensor[subject]              # [10]
        
        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)  # [16]

        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0)  # Shape: (1, 16)
        )
        data.subject_id = subject  # Track subject

        # === Store graph
        graph_data_list_addecode.append(data)
        final_subjects_with_all_data.append(subject)
        
        # === Print one example to verify shapes and content
        if len(graph_data_list_addecode) == 1:
            print("\n Example PyTorch Geometric Data object:")
            print("→ Node features shape:", data.x.shape)           # Ecpected: [84, 4]
            print("→ Edge index shape:", data.edge_index.shape)     # Ecpected: [2, ~3500]
            print("→ Edge attr shape:", data.edge_attr.shape)       # Ecpected: [~3500]
            print("→ Global features shape:", data.global_features.shape)  # Ecpected: [1, 16]
            print("→ Target age (y):", data.y.item())


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")







# ==== BLOCK 5.1: TRAIN AND EVALUATE FUNCTIONS ====

 
    
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Usamos el DataLoader de torch_geometric

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)  # GPU
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # GPU
            output = model(data).view(-1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)




adni_ckpt = torch.load(adni_model_path, map_location=device)
adni_state_dict = adni_ckpt["model_state_dict"]


# ==== BLOCK 5: TRANSFER LEARNING TRAINING LOOP ON AD-DECODE ====



import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import numpy as np

# Training parameters
epochs = 300
patience = 40  # Early stopping

k =  7 # Folds
batch_size = 6

# === Initialize losses ===
all_train_losses = []
all_test_losses = []

all_early_stopping_epochs = []  





#Age bins 


# === Extract subject IDs from graph data
graph_subject_ids = [data.subject_id for data in graph_data_list_addecode]

# === Filter and sort metadata to match only graph subjects
df_filtered = addecode_healthy_metadata_pca[
    addecode_healthy_metadata_pca["MRI_Exam_fixed"].isin(graph_subject_ids)
].copy()

# Double-check: remove any unexpected mismatches
df_filtered = df_filtered.drop_duplicates(subset="MRI_Exam_fixed", keep="first")
df_filtered = df_filtered.set_index("MRI_Exam_fixed")
df_filtered = df_filtered.loc[df_filtered.index.intersection(graph_subject_ids)]
df_filtered = df_filtered.loc[graph_subject_ids].reset_index()

# Final check
print(" Final matched lengths:")
print("  len(graphs):", len(graph_data_list_addecode))
print("  len(metadata):", len(df_filtered))

# === Extract final age vector and compute age bins
ages = df_filtered["age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)

print(" Aligned bins:", len(age_bins))








# Stratified split by age bins
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


repeats_per_fold = 10  


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):

    print(f'\n--- Fold {fold+1}/{k} ---')

    train_data = [graph_data_list_addecode[i] for i in train_idx]
    test_data = [graph_data_list_addecode[i] for i in test_idx]

    fold_train_losses = []
    fold_test_losses = []

    for repeat in range(repeats_per_fold):
        print(f'  > Repeat {repeat+1}/{repeats_per_fold}')
        
        early_stop_epoch = None  

        seed_everything(42 + repeat)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)






         # === LOAD MODEL + PRETRAINED WEIGHTS ===
        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        model.load_state_dict(adni_state_dict)
        for p in model.parameters():
            p.requires_grad = True




        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        best_loss = float('inf')
        patience_counter = 0

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                model_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"finetuned_from_adni1_fold_{fold+1}_rep_{repeat+1}.pt"
                    )

                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_epoch = epoch + 1  
                    print(f"    Early stopping triggered at epoch {early_stop_epoch}.")  
                    break


            scheduler.step()

        if early_stop_epoch is None:
                early_stop_epoch = epochs  
        all_early_stopping_epochs.append((fold + 1, repeat + 1, early_stop_epoch))


        fold_train_losses.append(train_losses)
        fold_test_losses.append(test_losses)

    all_train_losses.append(fold_train_losses)
    all_test_losses.append(fold_test_losses)
    
    
    
    



#################  LEARNING CURVE GRAPH (MULTIPLE REPEATS)  ################

plt.figure(figsize=(10, 6))

# Plot average learning curves across all repeats for each fold
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses[fold][rep], label=f'Train Loss - Fold {fold+1} Rep {rep+1}', linestyle='dashed', alpha=0.5)
        plt.plot(all_test_losses[fold][rep], label=f'Test Loss - Fold {fold+1} Rep {rep+1}', alpha=0.5)

plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curve (All Repeats)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)
plt.show()


# ==== LEARNING CURVE PLOT (MEAN ± STD) ====

import numpy as np
import matplotlib.pyplot as plt

# Compute mean and std for each epoch across all folds and repeats
avg_train = []
avg_test = []

for epoch in range(epochs):
    epoch_train = []
    epoch_test = []
    for fold in range(k):
        for rep in range(repeats_per_fold):
            if epoch < len(all_train_losses[fold][rep]):
                epoch_train.append(all_train_losses[fold][rep][epoch])
                epoch_test.append(all_test_losses[fold][rep][epoch])
    avg_train.append((np.mean(epoch_train), np.std(epoch_train)))
    avg_test.append((np.mean(epoch_test), np.std(epoch_test)))

# Unpack into arrays
train_mean, train_std = zip(*avg_train)
test_mean, test_std = zip(*avg_test)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(train_mean, label="Train Mean", color="blue")
plt.fill_between(range(epochs), np.array(train_mean) - np.array(train_std),
                 np.array(train_mean) + np.array(train_std), color="blue", alpha=0.3)

plt.plot(test_mean, label="Test Mean", color="orange")
plt.fill_between(range(epochs), np.array(test_mean) - np.array(test_std),
                 np.array(test_mean) + np.array(test_std), color="orange", alpha=0.3)

plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curve (Mean ± Std Across All Folds/Repeats)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#####################  PREDICTION & METRIC ANALYSIS ACROSS FOLDS/REPEATS  #####################


from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === Initialize storage ===
fold_mae_list = []
fold_r2_list = []
all_y_true = []
all_y_pred = []


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_data = [graph_data_list_addecode[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = BrainAgeGATv2(global_feat_dim=16).to(device)  

        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"finetuned_from_adni1_fold_{fold+1}_rep_{rep+1}.pt"
            )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))# Load correct model
        model.eval()

       

        y_true_repeat = []
        y_pred_repeat = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data).view(-1)
                y_pred_repeat.extend(output.cpu().tolist())
                y_true_repeat.extend(data.y.cpu().tolist())

        # Store values for this repeat
        mae = mean_absolute_error(y_true_repeat, y_pred_repeat)
        r2 = r2_score(y_true_repeat, y_pred_repeat)
        repeat_maes.append(mae)
        repeat_r2s.append(r2)

        all_y_true.extend(y_true_repeat)
        all_y_pred.extend(y_pred_repeat)

    fold_mae_list.append(repeat_maes)
    fold_r2_list.append(repeat_r2s)

    print(f">> Fold {fold+1} | MAE: {np.mean(repeat_maes):.2f} ± {np.std(repeat_maes):.2f} | R²: {np.mean(repeat_r2s):.2f} ± {np.std(repeat_r2s):.2f}")

# === Final aggregate results ===
all_maes = np.array(fold_mae_list).flatten()
all_r2s = np.array(fold_r2_list).flatten()

print("\n================== FINAL METRICS ==================")
print(f"Global MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f}")
print(f"Global R²:  {np.mean(all_r2s):.2f} ± {np.std(all_r2s):.2f}")
print("===================================================")



###############################################################
# PREDICTION & METRIC ANALYSIS — AD-DECODE  (SAVE TO CSV)
###############################################################
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

# ---------------------------  CONFIG  ---------------------------


BATCH_SIZE       = 6                                 # must match training
REPEATS_PER_FOLD = 10
N_FOLDS          = 7                                 # k en tu entrenamiento

# ---------------------------------------------------------------
# 1)  splits used in training
# ---------------------------------------------------------------
ages = np.array([data.y.item() for data in graph_data_list_addecode])  # 71 elementos
age_bins = pd.qcut(ages, q=5, labels=False)                            # 71 elementos

skf_addecode = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True, random_state=42)

# ---------------------------------------------------------------
# 2) lists
# ---------------------------------------------------------------
fold_mae, fold_rmse, fold_r2       = [], [], []
all_y_true, all_y_pred             = [], []
all_subject_ids, fold_tags         = [], []
repeat_tags                        = []

# ---------------------------------------------------------------
# 3) Loop per fold × repeat
# ---------------------------------------------------------------
for fold, (train_idx, test_idx) in enumerate(skf_addecode.split(
        graph_data_list_addecode, age_bins)):

    print(f"\n--- Evaluating AD-DECODE Fold {fold+1}/{N_FOLDS} ---")
    test_loader = DataLoader(
        [graph_data_list_addecode[i] for i in test_idx],
        batch_size=BATCH_SIZE, shuffle=False)

    mae_rep, rmse_rep, r2_rep = [], [], []            # métricas por repeat

    for rep in range(REPEATS_PER_FOLD):
        print(f"  > Repeat {rep+1}/{REPEATS_PER_FOLD}")

        # ----- Load trained model -----
        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"finetuned_from_adni1_fold_{fold+1}_rep_{rep+1}.pt"
            )

        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        # ----- Predictions -----
        y_true, y_pred, subj_ids = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch).view(-1)           # predicted age
                trues = batch.y.view(-1)                # real age

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(trues.cpu().tolist())
                subj_ids.extend([str(s) for s in batch.subject_id])

        # ----- Metrics -----
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        mae_rep.append(mae)
        rmse_rep.append(rmse)
        r2_rep.append(r2)

        # ----- Save in lists -----
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_subject_ids.extend(subj_ids)
        fold_tags.extend([fold+1] * len(y_true))
        repeat_tags.extend([rep+1] * len(y_true))

    # ----- Summary per fold -----
    fold_mae.append(mae_rep)
    fold_rmse.append(rmse_rep)
    fold_r2.append(r2_rep)

    print(f">> Fold {fold+1} | "
          f"MAE:  {np.mean(mae_rep):.2f} ± {np.std(mae_rep):.2f} | "
          f"RMSE: {np.mean(rmse_rep):.2f} ± {np.std(rmse_rep):.2f} | "
          f"R²:   {np.mean(r2_rep):.2f} ± {np.std(r2_rep):.2f}")

# ---------------------------------------------------------------
# 4) Global metrics
# ---------------------------------------------------------------
all_mae  = np.concatenate(fold_mae)
all_rmse = np.concatenate(fold_rmse)
all_r2   = np.concatenate(fold_r2)

print("\n================== FINAL METRICS AD-DECODE ==================")
print(f"Global MAE:  {all_mae.mean():.2f} ± {all_mae.std():.2f}")
print(f"Global RMSE: {all_rmse.mean():.2f} ± {all_rmse.std():.2f}")
print(f"Global R²:   {all_r2.mean():.2f} ± {all_r2.std():.2f}")
print("=============================================================\n")

# ---------------------------------------------------------------
# 5) Save CSV with all predictions
# ---------------------------------------------------------------
df_preds_adni_addecode = pd.DataFrame({
    "Subject_ID":    all_subject_ids,
    "Real_Age":      all_y_true,
    "Predicted_Age": all_y_pred,
    "Fold":          fold_tags,
    "Repeat":        repeat_tags
})

csv_path = os.path.join(OUT_DIR, "cv_predictions_addecode_pretrained_with_ADNImodel_FINETUNED1.csv")
df_preds_adni_addecode.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

# ============================================================
# A) BIAS CORRECTION (cBAG) PARA AD-DECODE
# - Ajusta bias SOLO con TRAIN (por fold y repeat)
# - Aplica corrección en TEST
# - Guarda CSV con pred_raw y pred_corr
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# --------------------------- CONFIG ---------------------------
BATCH_SIZE       = batch_size          # usa el mismo batch_size que arriba (6)
REPEATS_PER_FOLD = repeats_per_fold    # 10
N_FOLDS          = k                   # 7
# -------------------------------------------------------------

# age bins (usa las edades de graph_data_list_addecode)
ages = np.array([data.y.item() for data in graph_data_list_addecode], dtype=float)
age_bins = pd.qcut(ages, q=5, labels=False)

skf_addecode = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

rows = []
coef_rows = []

mae_raw_list, rmse_raw_list, r2_raw_list = [], [], []
mae_cor_list, rmse_cor_list, r2_cor_list = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf_addecode.split(graph_data_list_addecode, age_bins)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    train_loader = DataLoader([graph_data_list_addecode[i] for i in train_idx],
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader([graph_data_list_addecode[i] for i in test_idx],
                              batch_size=BATCH_SIZE, shuffle=False)

    for rep in range(REPEATS_PER_FOLD):
        print(f"  > Repeat {rep+1}/{REPEATS_PER_FOLD}")

        # ---- Carga modelo fine-tuned de ese fold/rep ----
        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"finetuned_from_adni1_fold_{fold+1}_rep_{rep+1}.pt"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # ============================================================
        # 1) PREDICT TRAIN -> FIT bias: BAG = pred - age  ~  age
        # ============================================================
        y_true_train, y_pred_train = [], []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                preds = model(batch).view(-1)
                trues = batch.y.view(-1)
                y_pred_train.extend(preds.detach().cpu().tolist())
                y_true_train.extend(trues.detach().cpu().tolist())

        age_train = np.array(y_true_train, dtype=float)
        pred_train = np.array(y_pred_train, dtype=float)

        bag_train = pred_train - age_train

        reg = LinearRegression().fit(age_train.reshape(-1, 1), bag_train)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)
        coef_rows.append({"Fold": fold+1, "Repeat": rep+1, "Slope": slope, "Intercept": intercept})

        # ============================================================
        # 2) PREDICT TEST -> APPLY correction
        #    cBAG = BAG - (slope*age + intercept)
        #    pred_corr = age + cBAG
        # ============================================================
        y_true, y_pred, subj_ids = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch).view(-1)
                trues = batch.y.view(-1)
                y_pred.extend(preds.detach().cpu().tolist())
                y_true.extend(trues.detach().cpu().tolist())
                subj_ids.extend([str(s) for s in batch.subject_id])

        age_test = np.array(y_true, dtype=float)
        pred_test = np.array(y_pred, dtype=float)

        bag_test = pred_test - age_test
        bias_hat = reg.predict(age_test.reshape(-1, 1))          # slope*age + intercept
        cbag_test = bag_test - bias_hat
        pred_corr = age_test + cbag_test

        # ---- métricas raw ----
        mae_raw = mean_absolute_error(age_test, pred_test)
        rmse_raw = rmse_np(age_test, pred_test)
        r2_raw = r2_score(age_test, pred_test)

        # ---- métricas corrected ----
        mae_cor = mean_absolute_error(age_test, pred_corr)
        rmse_cor = rmse_np(age_test, pred_corr)
        r2_cor = r2_score(age_test, pred_corr)

        mae_raw_list.append(mae_raw); rmse_raw_list.append(rmse_raw); r2_raw_list.append(r2_raw)
        mae_cor_list.append(mae_cor); rmse_cor_list.append(rmse_cor); r2_cor_list.append(r2_cor)

        # ---- guardar filas ----
        for sid, a, p, pc, b, cb in zip(subj_ids, age_test, pred_test, pred_corr, bag_test, cbag_test):
            rows.append({
                "Subject_ID": sid,
                "Real_Age": float(a),
                "Predicted_Age_raw": float(p),
                "Predicted_Age_corrected": float(pc),
                "BAG": float(b),
                "cBAG": float(cb),
                "Bias_Slope": slope,
                "Bias_Intercept": intercept,
                "Fold": fold+1,
                "Repeat": rep+1
            })

print("\n================== FINAL METRICS (RAW) ==================")
print(f"Global MAE:  {np.mean(mae_raw_list):.2f} ± {np.std(mae_raw_list):.2f}")
print(f"Global RMSE: {np.mean(rmse_raw_list):.2f} ± {np.std(rmse_raw_list):.2f}")
print(f"Global R²:   {np.mean(r2_raw_list):.2f} ± {np.std(r2_raw_list):.2f}")

print("\n============= FINAL METRICS (BIAS-CORRECTED) =============")
print(f"Global MAE:  {np.mean(mae_cor_list):.2f} ± {np.std(mae_cor_list):.2f}")
print(f"Global RMSE: {np.mean(rmse_cor_list):.2f} ± {np.std(rmse_cor_list):.2f}")
print(f"Global R²:   {np.mean(r2_cor_list):.2f} ± {np.std(r2_cor_list):.2f}")
print("=========================================================\n")

df_preds = pd.DataFrame(rows)
csv_out = os.path.join(OUT_DIR, "cv_predictions_addecode_bias_corrected_foldwise.csv")
df_preds.to_csv(csv_out, index=False)
print("Saved:", csv_out)

# ============================================================



from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# =========================
# PRINT RESULTS (LIKE BEFORE)
# =========================

# 1) Global correlations (OOF across folds×repeats)
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# --- SUBJECT-LEVEL aggregation (OOF mean per subject) ---
df_subj = (
    df_preds.groupby("Subject_ID", as_index=False)
            .agg(
                Age=("Real_Age", "first"),
                BAG=("BAG", "mean"),
            )
)

# --- Fit global bias on subject-level OOF: BAG ~ Age ---
reg_g = LinearRegression().fit(df_subj[["Age"]], df_subj["BAG"])
df_subj["cBAG_global"] = df_subj["BAG"] - reg_g.predict(df_subj[["Age"]])

# --- Correlations at SUBJECT-LEVEL (this is the one you want ~0) ---
r_bag, p_bag = pearsonr(df_subj["Age"].to_numpy(float), df_subj["BAG"].to_numpy(float))
r_cbg, p_cbg = pearsonr(df_subj["Age"].to_numpy(float), df_subj["cBAG_global"].to_numpy(float))

print("\n=========== BIAS CHECK (GLOBAL OOF, SUBJECT-LEVEL) ===========")
print(f"slope_global: {float(reg_g.coef_[0]):+.4f}")
print(f"intercept_global: {float(reg_g.intercept_):+.4f}")
print(f"corr(Age, BAG)        = {r_bag:+.4f} (p={p_bag:.2e})")
print(f"corr(Age, cBAG_global)= {r_cbg:+.4f} (p={p_cbg:.2e})")
print("==============================================================\n")

# --- map cBAG_global back to EVERY row (fold×repeat rows) ---
map_cbg = df_subj.set_index("Subject_ID")["cBAG_global"]
df_preds["cBAG_global"] = df_preds["Subject_ID"].map(map_cbg)

# optional: also store globally-corrected predicted age per row
df_preds["Predicted_Age_global_corrected"] = df_preds["Real_Age"] + df_preds["cBAG_global"]

# overwrite CSV so plots use it
df_preds.to_csv(csv_out, index=False)

# 2) Optional: per fold × repeat (to see if any split is weird)
print("================= BIAS CHECK (PER FOLD×REPEAT) =================")
grp = df_preds.groupby(["Fold", "Repeat"], as_index=False)

for (fold, rep), g in grp:
    a = g["Real_Age"].to_numpy(dtype=float)
    b = g["BAG"].to_numpy(dtype=float)
    c = g["cBAG"].to_numpy(dtype=float)

    # pearsonr needs >=2 points and non-constant vectors
    if len(a) < 2 or np.std(b) == 0 or np.std(c) == 0:
        print(f"Fold {fold} Rep {rep}: not enough variation to compute Pearson r")
        continue

    rb, pb = pearsonr(a, b)
    rc, pc = pearsonr(a, c)

    print(f"Fold {fold:>2} Rep {rep:>2} | corr(Age,BAG)={rb:+.3f} (p={pb:.1e}) "
          f"| corr(Age,cBAG)={rc:+.3f} (p={pc:.1e})")
print("=================================================================\n")




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

# ============================
# LOAD BIAS-CORRECTED PREDICTIONS
# ============================
dfc = pd.read_csv(csv_out)



PLOTS_DIR = os.path.join(OUT_DIR, "plots_bias_corrected")
os.makedirs(PLOTS_DIR, exist_ok=True)

from scipy.stats import pearsonr, linregress
import numpy as np
import matplotlib.pyplot as plt



def add_stats_box(ax, x, y, title_prefix=""):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    err = y - x
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    ss_res = float(np.sum((y - x)**2))
    ss_tot = float(np.sum((x - np.mean(x))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    r, p = pearsonr(x, y)
    lr = linregress(x, y)

    txt = (
        f"{title_prefix}"
        f"Pearson r = {r:+.3f} (p={p:.2e})\n"
        f"MAE = {mae:.2f}   RMSE = {rmse:.2f}   R² = {r2:.3f}\n"
        
    )

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="k")
    )


# ============================
# 1) OOF Pred vs Real (bias-corrected) + stats + error bars
# ============================

# Agrupar por sujeto para tener un punto por sujeto
# y usar la desviación estándar entre repeticiones como barra de error
df_plot = (
    dfc.groupby("Subject_ID", as_index=False)
       .agg(
           Real_Age=("Real_Age", "first"),
           Pred_Mean=("Predicted_Age_global_corrected", "mean"),
           Pred_STD=("Predicted_Age_global_corrected", "std")
       )
)

# Si algún sujeto solo tiene un valor, std será NaN -> poner 0
df_plot["Pred_STD"] = df_plot["Pred_STD"].fillna(0.0)

x = df_plot["Real_Age"].to_numpy(dtype=float)
y = df_plot["Pred_Mean"].to_numpy(dtype=float)
yerr = df_plot["Pred_STD"].to_numpy(dtype=float)

# Estadísticas
err = y - x
mae = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err**2)))
ss_res = float(np.sum((y - x)**2))
ss_tot = float(np.sum((x - np.mean(x))**2))
r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

r, p = pearsonr(x, y)
lr = linregress(x, y)

# Imprimir por consola
print("\n================ PREDICTED VS REAL (BIAS-CORRECTED) =================")
print(f"N subjects:      {len(df_plot)}")
print(f"Pearson r:       {r:+.4f}")
print(f"p-value:         {p:.4e}")
print(f"MAE:             {mae:.4f}")
print(f"RMSE:            {rmse:.4f}")
print(f"R²:              {r2:.4f}")
print(f"Fit slope:       {lr.slope:.4f}")
print(f"Fit intercept:   {lr.intercept:.4f}")
print("=====================================================================\n")

fig, ax = plt.subplots(figsize=(8,6))

# Puntos con barras de error
ax.errorbar(
    x, y, yerr=yerr,
    fmt='o',
    alpha=0.8,
    ecolor='gray',
    elinewidth=1.2,
    capsize=3,
    markersize=6,
    markeredgecolor='k'
)

mn = float(min(x.min(), y.min()))
mx = float(max(x.max(), y.max()))
xx = np.linspace(mn, mx, 200)

# Línea perfecta: y = x
ax.plot(
    xx, xx,
    linestyle="dashed",
    linewidth=2,
    color="red",
    label="Perfect agreement (y=x)"
)

# Línea del ajuste actual de los datos
ax.plot(
    xx, lr.slope * xx + lr.intercept,
    linestyle="-",
    linewidth=2,
    color="blue",
    label=f"Fit: y={lr.slope:.2f}x+{lr.intercept:.2f}"
)

ax.set_xlabel("Real Age")
ax.set_ylabel("Predicted Age (bias-corrected)")
ax.set_title("Predicted vs Real (subject mean ± std, bias-corrected)")
ax.grid(True)
ax.legend()

# Caja de estadísticas
txt = (
    f"Pearson r = {r:+.3f} (p={p:.2e})\n"
    f"MAE = {mae:.2f}   RMSE = {rmse:.2f}   R² = {r2:.3f}\n"
    f"Slope = {lr.slope:.3f}   Intercept = {lr.intercept:.3f}"
)

ax.text(
    0.02, 0.98, txt,
    transform=ax.transAxes,
    va="top", ha="left",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="k")
)

out_path = os.path.join(PLOTS_DIR, "pred_vs_real_oof_bias_corrected_with_errorbars_and_stats.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", out_path)

# ============================
# 2) Error vs Age (bias check) + stats (corr should be ~0)
# ============================
dfc["Error_corr"] = dfc["Predicted_Age_global_corrected"] - dfc["Real_Age"]
x2 = dfc["Real_Age"].to_numpy(dtype=float)
e2 = dfc["Error_corr"].to_numpy(dtype=float)

r_e, p_e = pearsonr(x2, e2)
lr_e = linregress(x2, e2)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(x2, e2, alpha=0.6, edgecolors="k")
ax.axhline(0, linestyle="dashed")
ax.set_xlabel("Real Age")
ax.set_ylabel("Error (Pred_corr - Real)")
ax.set_title("Error vs Age (bias-corrected)")
ax.grid(True)

txt = (
    f"r={r_e:+.3f} (p={p_e:.2e})\n"
   
)
ax.text(
    0.02, 0.98, txt,
    transform=ax.transAxes,
    va="top", ha="left",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="k")
)

out_path = os.path.join(PLOTS_DIR, "error_vs_age_bias_corrected_with_stats.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", out_path)

# ============================
# 3) SUBJECT-LEVEL mean across repeats + stats
# ============================
df_subj = (
    dfc.groupby("Subject_ID", as_index=False)
       .agg(Real_Age=("Real_Age","first"),
            Pred_corr=("Predicted_Age_global_corrected","mean"))
)

xs = df_subj["Real_Age"].to_numpy(dtype=float)
ys = df_subj["Pred_corr"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(xs, ys, alpha=0.8, edgecolors="k")
mn = float(min(xs.min(), ys.min()))
mx = float(max(xs.max(), ys.max()))
ax.plot([mn, mx], [mn, mx], linestyle="dashed")
ax.set_xlabel("Real Age")
ax.set_ylabel("Predicted Age (subject mean, bias-corrected)")
ax.set_title("Predicted vs Real (subject-level mean, bias-corrected)")
ax.grid(True)
add_stats_box(ax, xs, ys, title_prefix="SUBJECT MEAN\n")

out_path = os.path.join(PLOTS_DIR, "pred_vs_real_subject_mean_bias_corrected_with_stats.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", out_path)

print("\nAll bias-corrected plots saved in:", PLOTS_DIR)
df_coef = pd.DataFrame(coef_rows)
coef_out = os.path.join(OUT_DIR, "bias_coefficients_per_fold_repeat_addecode.csv")
df_coef.to_csv(coef_out, index=False)
print("Saved:", coef_out)





# ============================================================
# 2) Real vs Predicted (OOF scatter)

# ============================================================
# ============================
# RAW PLOTS (separate folder)
# ============================
PLOTS_DIR_RAW = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR_RAW, exist_ok=True)
print("Raw plots directory:", PLOTS_DIR_RAW)

def save_show_raw(fig, filename):
    path = os.path.join(PLOTS_DIR_RAW, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", path)
    
print("Plots directory:", PLOTS_DIR)
def save_show(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", path)
# Si ya tienes df_preds_adni_addecode en memoria, úsalo.
# Si no, descomenta para leer el CSV:
# df_preds_adni_addecode = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_addecode_pretrained_with_ADNImodel_FINETUNED1.csv"))

dfp = df_preds_adni_addecode.copy()
dfp["Error"] = dfp["Predicted_Age"] - dfp["Real_Age"]
dfp["AbsError"] = dfp["Error"].abs()
fig = plt.figure(figsize=(8,6))
plt.scatter(dfp["Real_Age"], dfp["Predicted_Age"], alpha=0.7, edgecolors="k")
mn = float(min(dfp["Real_Age"].min(), dfp["Predicted_Age"].min()))
mx = float(max(dfp["Real_Age"].max(), dfp["Predicted_Age"].max()))
plt.plot([mn, mx], [mn, mx], linestyle="dashed")
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real (OOF across folds×repeats)")
plt.grid(True)
save_show_raw(fig, "pred_vs_real_oof.png")

# ============================================================
from scipy.stats import pearsonr, linregress

# Error
dfp["Error"] = dfp["Predicted_Age"] - dfp["Real_Age"]

x = dfp["Real_Age"].to_numpy(dtype=float)
e = dfp["Error"].to_numpy(dtype=float)

# --- statistics ---
r, p = pearsonr(x, e)
lr = linregress(x, e)

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(x, e, alpha=0.6, edgecolors="k")
ax.axhline(0, linestyle="dashed")

ax.set_xlabel("Real Age")
ax.set_ylabel("Error (Pred - Real)")
ax.set_title("Error vs Age (raw)")
ax.grid(True)

# stats box
txt = (
    f"Pearson r = {r:.3f}\n"
    f"p-value = {p:.2e}\n"
    f"slope = {lr.slope:.3f}"
)

ax.text(
    0.02, 0.98,
    txt,
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="k")
)

out_path = os.path.join(PLOTS_DIR_RAW, "error_vs_age_raw_with_stats.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")

plt.show()
plt.close(fig)

# ============================================================
# 4) Bland–Altman: error vs mean(age, pred)
# ============================================================
mean_ap = 0.5 * (dfp["Real_Age"].values + dfp["Predicted_Age"].values)
diff = dfp["Predicted_Age"].values - dfp["Real_Age"].values
md = np.mean(diff)
sd = np.std(diff)

fig = plt.figure(figsize=(8,6))
plt.scatter(mean_ap, diff, alpha=0.7, edgecolors="k")
plt.axhline(md, linestyle="dashed")
plt.axhline(md + 1.96*sd, linestyle="dashed")
plt.axhline(md - 1.96*sd, linestyle="dashed")
plt.xlabel("Mean of (Real, Pred)")
plt.ylabel("Difference (Pred - Real)")
plt.title("Bland–Altman Plot")
plt.grid(True)
save_show_raw(fig, "bland_altman.png")

# ============================================================
# 5) Histogram of errors
# ============================================================
fig = plt.figure(figsize=(8,6))
plt.hist(dfp["Error"], bins=20, edgecolor="k", alpha=0.85)
plt.axvline(0, linestyle="dashed")
plt.xlabel("Error (Pred - Real)")
plt.ylabel("Count")
plt.title("Error distribution")
plt.grid(True)
save_show_raw(fig, "error_hist.png")

# ============================================================
# 6) Boxplot MAE per fold (stability)
# ============================================================
# MAE por fold y repeat
mae_by_fold_rep = dfp.groupby(["Fold", "Repeat"], as_index=False)["AbsError"].mean()

fig = plt.figure(figsize=(9,6))
folds_sorted = sorted(mae_by_fold_rep["Fold"].unique())
data = [mae_by_fold_rep.loc[mae_by_fold_rep["Fold"]==f, "AbsError"].values for f in folds_sorted]
plt.boxplot(data, labels=[str(f) for f in folds_sorted])
plt.xlabel("Fold")
plt.ylabel("MAE (mean abs error across test subjects)")
plt.title("MAE distribution per fold (across repeats)")
plt.grid(True)
save_show_raw(fig, "mae_boxplot_per_fold.png")

# ============================================================
# 7) Subject-level averaging (reduce repeats noise)
# ============================================================
df_subj = (
    dfp.groupby("Subject_ID", as_index=False)
       .agg(Real_Age=("Real_Age","first"),
            Predicted_Age=("Predicted_Age","mean"))
)
df_subj["Error"] = df_subj["Predicted_Age"] - df_subj["Real_Age"]

fig = plt.figure(figsize=(8,6))
plt.scatter(df_subj["Real_Age"], df_subj["Predicted_Age"], alpha=0.8, edgecolors="k")
mn = float(min(df_subj["Real_Age"].min(), df_subj["Predicted_Age"].min()))
mx = float(max(df_subj["Real_Age"].max(), df_subj["Predicted_Age"].max()))
plt.plot([mn, mx], [mn, mx], linestyle="dashed")
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (subject mean)")
plt.title("Predicted vs Real (subject-level mean across repeats)")
plt.grid(True)
save_show_raw(fig, "pred_vs_real_subject_mean.png")




mean_ap = 0.5 * (dfc["Real_Age"].values + dfc["Predicted_Age_global_corrected"].values)
diff = dfc["Predicted_Age_global_corrected"].values - dfc["Real_Age"].values

md = np.mean(diff)
sd = np.std(diff)

fig = plt.figure(figsize=(8,6))
plt.scatter(mean_ap, diff, alpha=0.7, edgecolors="k")
plt.axhline(md, linestyle="dashed")
plt.axhline(md + 1.96*sd, linestyle="dashed")
plt.axhline(md - 1.96*sd, linestyle="dashed")

plt.xlabel("Mean of (Real, Pred_corrected)")
plt.ylabel("Difference (Pred_corrected - Real)")
plt.title("Bland–Altman Plot (bias-corrected)")
plt.grid(True)

save_show(fig, "bland_altman_bias_corrected.png")
