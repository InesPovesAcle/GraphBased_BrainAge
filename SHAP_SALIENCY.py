#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:21:11 2026

@author: ines
"""

# ADDECODE 

    # Preprocess all risk data the same way as we did with healthy on previous script
    # Zscore only using healthy data !!
    # Then we use the trained model to predict age, BAG nd cBAG
  


#################  IMPORT NECESSARY LIBRARIES  ################


import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions

import torch
import random
import numpy as np

import networkx as nx  # For graph-level metrics

WORK = os.environ["WORK"]
RESULTS_DIR = os.path.join(WORK, "ines/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Set seed for reproducibility ===
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def save_and_show(fig_name):
    path = os.path.join(RESULTS_DIR, fig_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.show()

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
pca_path = os.path.join(WORK, "ines/data/PCA_human_blood_top30.csv")
df_pca = pd.read_csv(pca_path)
print(df_pca.head())

print(df_matched_connectomes.head())



# Fix id formats

# === Fix ID format in PCA DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE_1' → 'ADDECODE1'
df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)



# === Fix Subject format in metadata DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE1' → 'ADDECODE1'
df_matched_connectomes["IDRNA_fixed"] = df_matched_connectomes["IDRNA"].str.upper().str.replace("_", "", regex=False)




###### MATCH PCA GENES WITH METADATA############

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_withConnectome = df_matched_connectomes.merge(df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed")


#Numbers

# === Show how many subjects with PCA and connectome you have
print(f" subjects with metadata connectome: {df_matched_connectomes.shape[0]}")
print()

print(f" subjects with metadata PCA & connectome: {df_metadata_PCA_withConnectome.shape[0]}")
print()


# Get the full set of subject IDs (DWI_fixed) in set
all_ids = set(df_matched_connectomes["MRI_Exam_fixed"])

# Get the subject IDs (DWI_fixed) that matched with PCA
with_pca_ids = set(df_metadata_PCA_withConnectome["MRI_Exam_fixed"])

# Compute the difference:  subjects without PCA
without_pca_ids = all_ids - with_pca_ids

# Filter the original metadata for those subjects
df_without_pca = df_matched_connectomes[
    df_matched_connectomes["MRI_Exam_fixed"].isin(without_pca_ids)
]


# Print result
print(f" Subjects with connectome but NO PCA: {df_without_pca.shape[0]}")
print()

print(df_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]])






####################### FA MD Vol #############################
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
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume_norm.txt"
)

# === Load FA data ===

df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"]
df_fa = df_fa.reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col.startswith("S")]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)

# === Load MD data ===

df_md = pd.read_csv(md_path, sep="\t")
df_md = df_md[1:]
df_md = df_md[df_md["ROI"] != "0"]
df_md = df_md.reset_index(drop=True)
subject_cols_md = [col for col in df_md.columns if col.startswith("S")]
df_md_transposed = df_md[subject_cols_md].transpose()
df_md_transposed.columns = [f"ROI_{i+1}" for i in range(df_md_transposed.shape[1])]
df_md_transposed.index.name = "subject_id"
df_md_transposed = df_md_transposed.astype(float)

# === Load Volume data ===

df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"]
df_vol = df_vol.reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col.startswith("S")]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)


# === Combine FA + MD + Vol per subject ===
multimodal_features_dict = {}

for subj in df_fa_transposed.index:
    subj_id = subj.replace("S", "").zfill(5)
    if subj in df_md_transposed.index and subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float)
        stacked = torch.stack([fa, md, vol], dim=1)  # Shape: [84, 3]
        multimodal_features_dict[subj_id] = stacked

# === Normalization nodo-wise between subjects ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # [N_subjects, 84, 3]
    means = all_features.mean(dim=0)  # [84, 3]
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Apply normalization

# === Get IDs of healthy subjects only (exclude AD and MCI)
healthy_ids = df_matched_connectomes[~df_matched_connectomes["Risk"].isin(["AD", "MCI"])]["MRI_Exam_fixed"].tolist()

# === Stack node features only from healthy subjects
healthy_stack = torch.stack([
    multimodal_features_dict[subj]
    for subj in healthy_ids
    if subj in multimodal_features_dict
])  # Shape: [N_healthy, 84, 3]

# === Compute mean and std from healthy controls
node_means = healthy_stack.mean(dim=0)  # [84, 3]
node_stds = healthy_stack.std(dim=0) + 1e-8

# === Apply normalization to ALL subjects using healthy stats
normalized_node_features_dict = {
    subj: (features - node_means) / node_stds
    for subj, features in multimodal_features_dict.items()
}






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
def threshold_connectome(matrix, percentile=100):
    """
    Apply percentile-based thresholding to a connectome matrix.
    """
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

# --- Apply threshold + log transform ---
log_thresholded_connectomes = {}
for subject, matrix in matched_connectomes.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
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


# ===============================
# Step 11: Normalize Metadata and PCA Columns
# ===============================

from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

#label encoding sex
le_sex = LabelEncoder()
addecode_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_metadata_pca["sex"].astype(str))


# --- Label encode genotype ---
le = LabelEncoder()
addecode_metadata_pca["genotype"] = le.fit_transform(addecode_metadata_pca["genotype"].astype(str))

# --- Normalize numerical and PCA columns ---
numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3'] #Top 10 from SPEARMAN  corr (enrich)


# === Select healthy subjects only
df_controls = addecode_metadata_pca[~addecode_metadata_pca["Risk"].isin(["AD", "MCI"])]

# === Define all columns to z-score
all_zscore_cols = numerical_cols + pca_cols

# === Compute means and stds using only healthy controls
global_means = df_controls[all_zscore_cols].mean()
global_stds = df_controls[all_zscore_cols].std() + 1e-8

# === Apply normalization to ALL subjects using healthy stats
addecode_metadata_pca[all_zscore_cols] = (
    addecode_metadata_pca[all_zscore_cols] - global_means
) / global_stds




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
    for _, row in addecode_metadata_pca.iterrows()
}

# === 2. Graph metric tensor (clustering coefficient, path length) ===
subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_metadata_pca.iterrows()
}

# === 3. PCA tensor (top 10 age-correlated components) ===
subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_metadata_pca.iterrows()
}




#################  CONVERT MATRIX TO GRAPH  ################

graph_data_list_addecode = []
final_subjects_with_all_data = []  #verify subjects

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
        age_row = addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "age"
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




# Apply pretrained model


import torch
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

# === Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Define model architecture (must match the one used in training)
class BrainAgeGATv2(torch.nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()
        from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
        import torch.nn.functional as F
        import torch.nn as nn

        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True)
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
        x = self.gnn1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x + x_res3)

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

# === Load the final model trained on all healthy subjects
model = BrainAgeGATv2(global_feat_dim=16).to(device)
finalmodel_path = os.path.join(WORK, "ines/code/model_trained_on_all_healthy.pt")
model.eval()





# Predict on all addecode
from torch_geometric.loader import DataLoader

# === Build DataLoader using all processed subjects
loader = DataLoader(graph_data_list_addecode, batch_size=1, shuffle=False)

# === Collect predictions and ground-truth age
subject_ids = []
true_ages = []
predicted_ages = []

with torch.no_grad():
    for data in loader:
        data = data.to(device)
        output = model(data).item()  # Predicted age
        subject_ids.append(data.subject_id[0])
        true_ages.append(data.y.item())
        predicted_ages.append(output)




#Compute BAG and Corrected BAG (cBAG)

from sklearn.linear_model import LinearRegression
import pandas as pd

# === Create results DataFrame
df_preds = pd.DataFrame({
    "Subject_ID": subject_ids,
    "Age": true_ages,
    "Predicted_Age": predicted_ages
})

# === Compute raw BAG
df_preds["BAG"] = df_preds["Predicted_Age"] - df_preds["Age"]

# === Fit linear regression: BAG ~ Age (to capture systematic age bias)
reg = LinearRegression().fit(df_preds[["Age"]], df_preds["BAG"])

# === Correct the BAG by removing the age-related component
df_preds["cBAG"] = df_preds["BAG"] - reg.predict(df_preds[["Age"]])



# Save cvs

# === Align metadata (to include Risk, Sex, Genotype, etc.)
df_preds_aligned = df_preds.set_index("Subject_ID").loc[
    addecode_metadata_pca.set_index("MRI_Exam_fixed").index
].reset_index()

# === Add relevant metadata columns
meta_cols = ["Risk", "sex", "genotype", "APOE", "Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"] + pca_cols
for col in meta_cols:
    df_preds_aligned[col] = addecode_metadata_pca[col].values

# === Replace missing or blank 'Risk' values with 'NoRisk'
df_preds_aligned["Risk"] = df_preds_aligned["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)

# === Save to CSV for future analysis
output_path = "brain_age_predictions_with_metadata.csv"
df_preds_aligned.to_csv(output_path, index=False)
print(f"Saved: {output_path}")






# SHAP 


import shap
import torch

# === Wrapper to use only global features ===
class GlobalOnlyModel(torch.nn.Module):
    def __init__(self, original_model):
        super(GlobalOnlyModel, self).__init__()
        self.meta_head = original_model.meta_head
        self.graph_head = original_model.graph_head
        self.pca_head = original_model.pca_head
        self.fc = original_model.fc

    def forward(self, global_feats):
        meta_embed = self.meta_head(global_feats[:, 0:4])
        graph_embed = self.graph_head(global_feats[:, 4:6])
        pca_embed = self.pca_head(global_feats[:, 6:])
        global_embed = torch.cat([meta_embed, graph_embed, pca_embed], dim=1)
        x = torch.cat([torch.zeros(global_feats.size(0), 128).to(global_feats.device), global_embed], dim=1)
        x = self.fc(x)
        return x


# === Extract global features and subject IDs
global_feats = []
subject_ids = []

for data in graph_data_list_addecode:
    global_feats.append(data.global_features.squeeze(0))  # shape: [16]
    subject_ids.append(data.subject_id)

global_feats_tensor = torch.stack(global_feats).to(device)  # shape: [N, 16]

# === Inicialize model SHAP-ready
wrapped_model = GlobalOnlyModel(model).to(device)
wrapped_model.eval()

# === Use SHAP DeepExplainer
explainer = shap.DeepExplainer(wrapped_model, global_feats_tensor)
shap_values = explainer.shap_values(global_feats_tensor)


if isinstance(shap_values, list):
    shap_values = shap_values[0]

if hasattr(shap_values, "ndim") and shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values[:, :, 0]


# === Names of the 16 global features (must match model input order)
feature_names = (
    ["Systolic", "Diastolic", "sex_encoded", "genotype"] +
    ["Clustering_Coeff", "Path_Length"] +
    ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3']  # Top 10 by Spearman correlation
)

# === Create DataFrame from SHAP values
df_shap = pd.DataFrame(shap_values, columns=feature_names)
df_shap["Subject_ID"] = subject_ids

# === Add Age column using graph_data_list_addecode
subject_to_age = {
    data.subject_id: data.y.item()
    for data in graph_data_list_addecode
}
df_shap["Age"] = df_shap["Subject_ID"].map(subject_to_age)

# === Reorder columns: Subject_ID, Age, SHAP features
cols = ["Subject_ID", "Age"] + feature_names
df_shap = df_shap[cols]

# === Save to CSV for later analysis
df_shap.to_csv("shap_global_features_all_subjects.csv", index=False)
print("Saved: shap_global_features_all_subjects.csv")








import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Helper function to save
# =========================
def save_figure(filename):
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close()


# ==========================================================
# GLOBAL SHAP – Feature groups
# ==========================================================

df_shap = pd.read_csv("shap_global_features_all_subjects.csv")

demographic_cols = ["Systolic", "Diastolic", "sex_encoded", "genotype"]
graphmetric_cols = ["Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3']

def plot_beeswarm(df, features, title, filename):
    shap_matrix = df[features].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    save_figure(filename)

# Global plots
plot_beeswarm(df_shap, demographic_cols,
              "SHAP Beeswarm — Demographic Features",
              "shap_demographics.png")

plot_beeswarm(df_shap, graphmetric_cols,
              "SHAP Beeswarm — Connectome Graph Metrics",
              "shap_graph_metrics.png")

plot_beeswarm(df_shap, pca_cols,
              "SHAP Beeswarm — PCA Gene Components",
              "shap_pca_genes.png")


# ==========================================================
# SHAP BY AGE GROUP
# ==========================================================

df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3,
                               labels=["Young", "Middle", "Old"])

feature_names = df_shap.columns.difference(
    ["Subject_ID", "Age", "Age_Group"]
).tolist()

def plot_beeswarm_age(df_group, title, filename):
    shap_matrix = df_group[feature_names].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=feature_names
    )
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(feature_names), show=False)
    plt.title(title)
    plt.tight_layout()
    save_figure(filename)

for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    plot_beeswarm_age(df_group,
                      f"SHAP Beeswarm — Age Group: {group}",
                      f"shap_agegroup_{group}.png")


# ==========================================================
# SHAP BY AGE GROUP AND FEATURE TYPE
# ==========================================================

for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]

    plot_beeswarm(df_group, demographic_cols,
                  f"SHAP Demographics ({group})",
                  f"shap_demographics_{group}.png")

    plot_beeswarm(df_group, graphmetric_cols,
                  f"SHAP Graph Metrics ({group})",
                  f"shap_graphmetrics_{group}.png")

    plot_beeswarm(df_group, pca_cols,
                  f"SHAP PCA Genes ({group})",
                  f"shap_pca_{group}.png")


# ==========================================================
# INDIVIDUAL SUBJECT PLOTS
# ==========================================================

feature_names = df_shap.columns.difference(
    ["Subject_ID", "Age", "Age_Group"]
).tolist()

subject_young = df_shap.loc[df_shap["Age"].idxmin()]
subject_old = df_shap.loc[df_shap["Age"].idxmax()]
median_age = df_shap["Age"].median()
subject_middle = df_shap.iloc[
    (df_shap["Age"] - median_age).abs().argsort().iloc[0]
]

def plot_subject_shap_signed(subject_row, filename):
    subject_id = subject_row["Subject_ID"]
    age = subject_row["Age"]

    shap_values = subject_row[feature_names]
    shap_sorted = shap_values.reindex(
        shap_values.abs().sort_values(ascending=True).index
    )

    plt.figure(figsize=(7, 5))
    shap_sorted.plot(
        kind="barh",
        color=shap_sorted.apply(
            lambda x: "crimson" if x < 0 else "steelblue"
        )
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
    plt.xlabel("SHAP value (contribution to prediction)")
    plt.title(f"SHAP Impact — Subject {subject_id} (Age: {int(age)})")
    plt.tight_layout()
    save_figure(filename)

plot_subject_shap_signed(subject_young, "shap_subject_young.png")
plot_subject_shap_signed(subject_middle, "shap_subject_middle.png")
plot_subject_shap_signed(subject_old, "shap_subject_old.png")


# ==========================================================
# NODE-LEVEL BEESWARM (Top 20 regions)
# ==========================================================

df_shap_nodes = pd.read_csv("shap_node_feature_importance.csv")

shap_matrix = df_shap_nodes.drop(columns="Subject_ID").values
region_names = df_shap_nodes.columns[1:].tolist()

shap_values = shap.Explanation(
    values=shap_matrix,
    base_values=None,
    data=shap_matrix,
    feature_names=region_names
)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP Beeswarm Plot — Top 20 Brain Regions")
plt.tight_layout()
save_figure("shap_regions_top20.png")


####SALIENCY####
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Asegúrate de tener RESULTS_DIR definido como ya lo tienes ===
# RESULTS_DIR = os.path.join(WORK, "ines/results")

def saliency_node_features_one_subject(model, data, device, region_names, results_dir,
                                       tag="subject", use_abs=True):
    """
    Saliency por región basado en gradiente d(output)/d(x).
    Devuelve:
      - df_region: saliency por región (84,)
      - df_region_feat: saliency por región y feature (84,4)
      - guarda CSV + PNG
    """
    model.eval()

    # Copia el grafo al device
    data = data.to(device)

    # IMPORTANTE: activar gradientes en x
    data.x = data.x.clone().detach().requires_grad_(True)

    # (opcional) si quieres también grad en global_features:
    if hasattr(data, "global_features") and data.global_features is not None:
        data.global_features = data.global_features.clone().detach()  # no grad aquí

    # Forward
    out = model(data)  # shape [batch, 1] si batch_size=1
    score = out.squeeze()  # escalar

    # Backward
    model.zero_grad(set_to_none=True)
    if data.x.grad is not None:
        data.x.grad.zero_()
    score.backward()

    # Gradiente respecto a x
    grads = data.x.grad.detach().cpu().numpy()   # (84,4)

    # Saliency (a veces se usa abs del gradiente)
    if use_abs:
        grads = np.abs(grads)

    # Agrega por región: media o norma L1/L2; aquí uso media
    region_saliency = grads.mean(axis=1)  # (84,)
    region_feat_saliency = grads          # (84,4)

    # DataFrames
    feat_cols = ["FA", "MD", "Vol", "Clustering"]  # tus 4 features por nodo
    df_region_feat = pd.DataFrame(region_feat_saliency, columns=feat_cols)
    df_region_feat.insert(0, "Region", region_names)

    df_region = pd.DataFrame({
        "Region": region_names,
        "Saliency": region_saliency
    }).sort_values("Saliency", ascending=False).reset_index(drop=True)

    # Guardar CSV
    csv1 = os.path.join(results_dir, f"saliency_regions_{tag}.csv")
    csv2 = os.path.join(results_dir, f"saliency_regions_x_features_{tag}.csv")
    df_region.to_csv(csv1, index=False)
    df_region_feat.to_csv(csv2, index=False)

    # Plot top 20 regiones
    topN = 20
    df_top = df_region.head(topN).iloc[::-1]  # para barh

    plt.figure(figsize=(10, 7))
    plt.barh(df_top["Region"], df_top["Saliency"])
    plt.xlabel("Saliency (|dŷ/dx| agregado por región)")
    plt.title(f"Saliency por región (Top {topN}) — {tag}")
    plt.tight_layout()

    fig_path = os.path.join(results_dir, f"saliency_regions_top{topN}_{tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved:")
    print(" -", csv1)
    print(" -", csv2)
    print(" -", fig_path)

    return df_region, df_region_feat


# =========================
# EJEMPLO DE USO
# =========================
# region_names: tu lista de 84 nombres (ya la tienes definida en tu script)
# graph_data_list_addecode: tu lista de grafos
# model: tu modelo cargado
# device: torch.device(...)

# Elige un sujeto (por ejemplo el primero)
idx = 0
data0 = graph_data_list_addecode[idx]
tag = f"{getattr(data0, 'subject_id', 'subj')}"

df_region, df_region_feat = saliency_node_features_one_subject(
    model=model,
    data=data0,
    device=device,
    region_names=region_names,
    results_dir=RESULTS_DIR,
    tag=str(tag),
    use_abs=True
)

# def compute_average_saliency(model, data_list, device, region_names):
#     model.eval()

#     all_saliency = []

#     for data in data_list:
#         data = data.to(device)
#         data.x = data.x.clone().detach().requires_grad_(True)

#         output = model(data)
#         score = output.squeeze()

#         model.zero_grad()
#         score.backward()

#         grads = data.x.grad.detach().cpu().numpy()
#         grads = np.abs(grads)

#         region_saliency = grads.mean(axis=1)
#         all_saliency.append(region_saliency)

#     mean_saliency = np.mean(all_saliency, axis=0)

#     df_mean = pd.DataFrame({
#         "Region": region_names,
#         "Saliency": mean_saliency
#     }).sort_values(by="Saliency", ascending=False).reset_index(drop=True)

#     csv_path = os.path.join(RESULTS_DIR, "saliency_regions_average.csv")
#     df_mean.to_csv(csv_path, index=False)

#     # Plot
#     top_n = 20
#     df_top = df_mean.head(top_n).iloc[::-1]

#     plt.figure(figsize=(10, 7))
#     plt.barh(df_top["Region"], df_top["Saliency"])
#     plt.xlabel("Average Saliency")
#     plt.title(f"Average Saliency by Region (Top {top_n})")
#     plt.tight_layout()

#     fig_path = os.path.join(RESULTS_DIR, "saliency_regions_average_top20.png")
#     plt.savefig(fig_path, dpi=300)
#     plt.close()

#     print("Saved:")
#     print(csv_path)
#     print(fig_path)

#     return df_mean
def compute_average_saliency(model, data_list, device, region_names, err="sem"):
    """
    err: "sd" | "sem" | "ci95"
    """
    model.eval()
    all_saliency = []

    for data in data_list:
        data = data.to(device)
        data.x = data.x.clone().detach().requires_grad_(True)

        out = model(data).squeeze()
        model.zero_grad(set_to_none=True)
        if data.x.grad is not None:
            data.x.grad.zero_()
        out.backward()

        grads = data.x.grad.detach().cpu().numpy()
        grads = np.abs(grads)
        region_saliency = grads.mean(axis=1)  # (84,)
        all_saliency.append(region_saliency)

    all_saliency = np.stack(all_saliency, axis=0)  # (N,84)
    n = all_saliency.shape[0]

    mean_saliency = all_saliency.mean(axis=0)
    sd_saliency = all_saliency.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean_saliency)
    sem_saliency = sd_saliency / np.sqrt(n) if n > 0 else np.zeros_like(mean_saliency)

    if err == "sd":
        xerr = sd_saliency
        err_label = "SD"
    elif err == "sem":
        xerr = sem_saliency
        err_label = "SEM"
    elif err == "ci95":
        xerr = 1.96 * sem_saliency
        err_label = "95% CI"
    else:
        raise ValueError("err must be 'sd', 'sem', or 'ci95'")

    df_mean = pd.DataFrame({
        "Region": region_names,
        "MeanSaliency": mean_saliency,
        "SD": sd_saliency,
        "SEM": sem_saliency,
        "N": n
    }).sort_values(by="MeanSaliency", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(RESULTS_DIR, "saliency_regions_average_mean_sd_sem.csv")
    df_mean.to_csv(csv_path, index=False)

    # Plot con error bars
    top_n = 20
    df_top = df_mean.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(
        y=df_top["Region"],
        width=df_top["MeanSaliency"],
        xerr=df_top["SEM"] if err == "sem" else (df_top["SD"] if err == "sd" else 1.96 * df_top["SEM"]),
        capsize=4,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    plt.xlabel(f"Average Saliency ± {err_label} (n={n})")
    plt.title(f"Average Saliency by Region (Top {top_n})")
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, f"saliency_regions_average_top{top_n}_errorbars_{err}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved:")
    print(csv_path)
    print(fig_path)

    return df_mean

df_mean_saliency = compute_average_saliency(
    model=model,
    data_list=graph_data_list_addecode,
    device=device,
    region_names=region_names
)
df_mean_saliency = compute_average_saliency(
    model=model,
    data_list=graph_data_list_addecode,
    device=device,
    region_names=region_names,
    err="ci95"   # prueba CI95 para que se vean más
)
