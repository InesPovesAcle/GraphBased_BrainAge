#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:08:18 2026

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
from scipy.stats import linregress
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

WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
RESULTS_DIR = os.path.join(WORK, "ines/results/BrainAgePredictionAll_withoutPCA")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ADDECODE Data

####################### CONNECTOMES ###############################
print("ADDECODE CONNECTOMES\n")

# === Define paths ===
zip_path = os.path.join(WORK,
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
metadata_path = os.path.join(WORK,
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





addecode_metadata = df_matched_connectomes.reset_index(drop=True)



####################### FA MD Vol #############################



# === Load FA data ===
fa_path = os.path.join(WORK,
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

# === Load MD data ===
md_path = os.path.join(WORK,
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

# === Load Volume data ===
vol_path = os.path.join(WORK,
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
    thresholded_matrix = threshold_connectome(matrix, percentile=70)
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
addecode_metadata_pca = addecode_metadata.copy()  # (puedes renombrar la variable si quieres)
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

df_controls = addecode_metadata_pca[~addecode_metadata_pca["Risk"].isin(["AD", "MCI"])]

global_means = df_controls[numerical_cols].mean()
global_stds  = df_controls[numerical_cols].std() + 1e-8

addecode_metadata_pca[numerical_cols] = (
    addecode_metadata_pca[numerical_cols] - global_means
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
        
        
        global_feat = torch.cat([demo_tensor, graph_tensor], dim=0)  # [16]

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
    def __init__(self):
        super(BrainAgeGATv2, self).__init__()
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

        # 4 demográficas
        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # 2 métricas grafo
        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # input = 128 (GNN) + 32 (meta+graph)
        self.fc = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        import torch
        from torch_geometric.nn import global_mean_pool

        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x = self.node_embed(x)
        x = self.gnn1(x, edge_index, data.edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index, data.edge_attr)
        x = self.bn2(x)
        x = torch.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index, data.edge_attr)
        x = self.bn3(x)
        x = torch.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index, data.edge_attr)
        x = self.bn4(x)
        x = torch.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)

        global_feats = data.global_features.to(x.device).squeeze(1)  # (B,6)
        meta_embed  = self.meta_head(global_feats[:, 0:4])           # (B,16)
        graph_embed = self.graph_head(global_feats[:, 4:6])          # (B,16)

        global_embed = torch.cat([meta_embed, graph_embed], dim=1)   # (B,32)

        x = torch.cat([x, global_embed], dim=1)                      # (B,160)
        x = self.fc(x)
        return x
    
# === Load the final model trained on all healthy subjects
finalmodel_path = os.path.join(WORK,
    "ines/results/BrainAgePredictionHealthy_withoutPCA/model_trained_on_all_healthy_without_PCA.pt"
)

model = BrainAgeGATv2().to(device)
model.load_state_dict(torch.load(finalmodel_path))
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
meta_cols = ["Risk", "sex", "genotype",  "Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
for col in meta_cols:
    df_preds_aligned[col] = addecode_metadata_pca[col].values

# === Replace missing or blank 'Risk' values with 'NoRisk'
df_preds_aligned["Risk"] = df_preds_aligned["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)

# === Save to CSV for future analysis
output_path = os.path.join(RESULTS_DIR, "brain_age_predictions_all_without_PCA.csv")
df_preds_aligned.to_csv(output_path, index=False)
print(f"Saved: {output_path}")




from pathlib import Path

def save_fig(name, results_dir=RESULTS_DIR, dpi=300):
    """
    Guarda la figura actual (plt.gcf()) como PNG y PDF en RESULTS_DIR.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()

    png_path = os.path.join(results_dir, f"{name}.png")


    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    
    print(f"Saved figure: {png_path}")



#Visualize BAG and cBAG vs Age
import seaborn as sns
import matplotlib.pyplot as plt

# === BAG vs Age
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.scatterplot(data=df_preds, x="Age", y="BAG", alpha=0.6)
sns.regplot(data=df_preds, x="Age", y="BAG", scatter=False, color="red", label="Trend")
plt.axhline(0, linestyle="--", color="gray")
x = df_preds["Age"].values
y = df_preds["BAG"].values
mask = np.isfinite(x) & np.isfinite(y)
res = linregress(x[mask], y[mask])
r2 = res.rvalue**2
p = res.pvalue
ax.text(
    0.02, 0.98,
    f"$R^2$ = {r2:.3f}\n$p$ = {p:.3g}",
    transform=ax.transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
plt.title("BAG vs Age (Before Correction)")
plt.xlabel("Chronological Age")
plt.ylabel("Brain Age Gap (BAG)")
plt.legend()
plt.tight_layout()
save_fig("BAG_vs_Age_before_correction")
plt.show()

# === cBAG vs Age
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.scatterplot(data=df_preds, x="Age", y="cBAG", alpha=0.6)
sns.regplot(data=df_preds, x="Age", y="cBAG", scatter=False, color="green", label="Trend")
plt.axhline(0, linestyle="--", color="gray")
x = df_preds["Age"].values
y = df_preds["cBAG"].values
mask = np.isfinite(x) & np.isfinite(y)
res = linregress(x[mask], y[mask])
r2 = res.rvalue**2
p = res.pvalue
ax.text(
    0.02, 0.98,
    f"$R^2$ = {r2:.3f}\n$p$ = {p:.3g}",
    transform=ax.transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.title("Corrected BAG vs Age (After Correction)")
plt.xlabel("Chronological Age")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.legend()
plt.tight_layout()
save_fig("cBAG_vs_Age_after_correction")
plt.show()









# VIOLIN PLOT BAG-RISK


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load the data
csv_path = os.path.join(RESULTS_DIR, "brain_age_predictions_all_without_PCA.csv")
df = pd.read_csv(csv_path)





# === Define risk group order explicitly
risk_order = ["NoRisk", "Familial", "MCI", "AD"]

# === Violin plot: BAG vs Risk
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="Risk", y="BAG", order=risk_order, inner="box", palette="Set2")
plt.title("Brain Age Gap (BAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Brain Age Gap (BAG)")
plt.tight_layout()
save_fig("violin_BAG_by_Risk")
plt.show()

# === Violin plot: cBAG vs Risk
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="Risk", y="cBAG", order=risk_order, inner="box", palette="Set2")
plt.title("Corrected Brain Age Gap (cBAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.tight_layout()
save_fig("violin_cBAG_by_Risk")
plt.show()


summary_path = os.path.join(RESULTS_DIR, "summary_by_risk.csv")
df.groupby("Risk")[["BAG","cBAG","Age","Predicted_Age"]].agg(["count","mean","std","median"]).to_csv(summary_path)
print(f"Saved: {summary_path}")



