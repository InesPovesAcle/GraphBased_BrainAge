#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:42:14 2026
@author: ines

ADDECODE
- Match connectomes + metadata + PCA
- Build multimodal node features (FA, MD, Vol-mm3) normalized using controls only
- Build graphs + load pretrained model + predict
- Compute BAG + cBAG
- Read volume_norm (ROI0 mL, ROI17+ROI53 hippocampus %, read-only) and correlate/plot
- Uniform plots (same style, regression line + 95% CI) saved as PNGs under $WORK/ines/results
"""

import os
import re
import zipfile
import random

import numpy as np
import pandas as pd

import torch
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm


# ============================================================
# Global config
# ============================================================
WORK = os.environ["WORK"]
RESULTS_DIR = os.path.join(WORK, "ines/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set(style="whitegrid", context="talk")  # uniform style


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zfill5(x) -> str:
    return str(x).strip().zfill(5)


def col_to_sid(colname: str) -> str:
    """Extract 5-digit subject id from a column like 'S01912_master_T' -> '01912'."""
    m = re.search(r"S(\d+)", str(colname))
    return m.group(1).zfill(5) if m else str(colname)


def ensure_roi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a numeric 'ROI' column.
    Handles cases where ROI is in index or column, and where first row is header-like.
    """
    if "ROI" in df.columns:
        df = df.copy()
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
        return df
    # ROI might be index
    df2 = df.copy()
    df2 = df2.reset_index()
    if "ROI" not in df2.columns:
        # assume first column is ROI
        df2 = df2.rename(columns={df2.columns[0]: "ROI"})
    df2["ROI"] = pd.to_numeric(df2["ROI"], errors="coerce")
    return df2


def regplot_and_save(df: pd.DataFrame, x: str, y: str, title: str, out_png: str, add_hline0=False):
    """Uniform scatter + regression line + 95% CI + Pearson r,p in title; save to PNG."""
    d = df.dropna(subset=[x, y]).copy()
    if len(d) < 3:
        print(f"[WARN] Not enough points to plot {y} vs {x} (n={len(d)}). Skipping:", out_png)
        return

    r, p = pearsonr(d[x], d[y])

    plt.figure(figsize=(7, 6))
    sns.regplot(
        data=d,
        x=x,
        y=y,
        ci=95,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "black"},
    )
    if add_hline0:
        plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{title}\nr = {r:.3f}, p = {p:.4g}, n = {len(d)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print("Saved:", out_png)


# ============================================================
# Step 1: Load connectomes
# ============================================================
print("ADDECODE CONNECTOMES\n")

zip_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/connectomes/AD_DECODE_connectome_act.zip",
)
directory_inside_zip = "connectome_act/"
connectomes = {}

with zipfile.ZipFile(zip_path, "r") as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                mat = pd.read_csv(f, header=None)
                fname = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[fname] = mat

print(f"Total connectome matrices loaded: {len(connectomes)}")

# filter out white matter
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# clean to 5-digit IDs from 'Sxxxxx...'
cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    m = re.search(r"S(\d+)", k)
    if m:
        sid = m.group(1).zfill(5)
        cleaned_connectomes[sid] = v

print()


# ============================================================
# Step 2: Load metadata
# ============================================================
print("ADDECODE METADATA\n")

metadata_path = os.path.join(WORK, "ines/data/AD_DECODE_data4.xlsx")
df_metadata = pd.read_excel(metadata_path)

df_metadata = df_metadata.dropna(how="all").copy()
df_metadata = df_metadata.dropna(subset=["MRI_Exam"]).copy()
df_metadata["MRI_Exam_fixed"] = df_metadata["MRI_Exam"].apply(zfill5)

print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print()


# ============================================================
# Step 3: Match connectomes & metadata
# ============================================================
print("MATCHING CONNECTOMES WITH METADATA")

matched_metadata = df_metadata[df_metadata["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())].copy()
print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}
df_matched_connectomes = matched_metadata.copy()


# ============================================================
# Step 4: Load PCA genes + match
# ============================================================
print("PCA GENES")

pca_path = os.path.join(WORK, "ines/data/PCA_human_blood_top30.csv")
df_pca = pd.read_csv(pca_path)

df_pca["ID_fixed"] = df_pca["ID"].astype(str).str.upper().str.replace("_", "", regex=False)
df_matched_connectomes["IDRNA_fixed"] = df_matched_connectomes["IDRNA"].astype(str).str.upper().str.replace("_", "", regex=False)

df_metadata_PCA_withConnectome = df_matched_connectomes.merge(
    df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed"
)

print(f"subjects with metadata+connectome: {df_matched_connectomes.shape[0]}")
print(f"subjects with metadata+PCA+connectome: {df_metadata_PCA_withConnectome.shape[0]}")

all_ids = set(df_matched_connectomes["MRI_Exam_fixed"])
with_pca_ids = set(df_metadata_PCA_withConnectome["MRI_Exam_fixed"])
without_pca_ids = all_ids - with_pca_ids
print(f"Subjects with connectome but NO PCA: {len(without_pca_ids)}\n")


# ============================================================
# Step 5: Load FA/MD/Vol (RAW mm3 for node features)
# ============================================================
print("FA / MD / VOL (node features)\n")

fa_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt",
)
md_path = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt",
)
vol_path_raw = os.path.join(
    WORK,
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume_norm.txt",
)

# FA
df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = ensure_roi_column(df_fa)
df_fa = df_fa[df_fa["ROI"] != 0].copy()
subject_cols_fa = [c for c in df_fa.columns if str(c).startswith("S")]
df_fa_t = df_fa[subject_cols_fa].transpose()
df_fa_t = df_fa_t.apply(pd.to_numeric, errors="coerce").astype(float)
df_fa_t.index.name = "subject_id"

# MD
df_md = pd.read_csv(md_path, sep="\t")
df_md = ensure_roi_column(df_md)
df_md = df_md[df_md["ROI"] != 0].copy()
subject_cols_md = [c for c in df_md.columns if str(c).startswith("S")]
df_md_t = df_md[subject_cols_md].transpose()
df_md_t = df_md_t.apply(pd.to_numeric, errors="coerce").astype(float)
df_md_t.index.name = "subject_id"

# Vol raw (mm3)
df_vol_raw = pd.read_csv(vol_path_raw, sep="\t")
df_vol_raw = ensure_roi_column(df_vol_raw)
df_vol_raw = df_vol_raw[df_vol_raw["ROI"] != 0].copy()
subject_cols_vol = [c for c in df_vol_raw.columns if str(c).startswith("S")]
df_vol_t = df_vol_raw[subject_cols_vol].transpose()
df_vol_t = df_vol_t.apply(pd.to_numeric, errors="coerce").astype(float)
df_vol_t.index.name = "subject_id"

# Build multimodal node features dict
multimodal_features_dict = {}
for subj in df_fa_t.index:
    sid = col_to_sid(subj)
    if subj in df_md_t.index and subj in df_vol_t.index:
        fa = torch.tensor(df_fa_t.loc[subj].values, dtype=torch.float)
        md = torch.tensor(df_md_t.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_t.loc[subj].values, dtype=torch.float)  # mm3
        stacked = torch.stack([fa, md, vol], dim=1)  # [nROI,3]
        multimodal_features_dict[sid] = stacked


# ============================================================
# Step 6: Normalize node-wise features using healthy controls only
# ============================================================
healthy_ids = df_matched_connectomes.loc[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"]),
    "MRI_Exam_fixed",
].astype(str).str.zfill(5).tolist()

healthy_stack_list = [multimodal_features_dict[s] for s in healthy_ids if s in multimodal_features_dict]
if len(healthy_stack_list) == 0:
    raise ValueError("No healthy subjects found with multimodal node features. Check Risk labels + IDs.")
healthy_stack = torch.stack(healthy_stack_list)  # [N_healthy, nROI, 3]

node_means = healthy_stack.mean(dim=0)
node_stds = healthy_stack.std(dim=0) + 1e-8

normalized_node_features_dict = {
    sid: (feat - node_means) / node_stds
    for sid, feat in multimodal_features_dict.items()
}


# ============================================================
# Step 7: Graph metrics
# ============================================================
def compute_nodewise_clustering_coefficients(matrix: pd.DataFrame) -> torch.Tensor:
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = float(matrix.iloc[u, v])
    cc = nx.clustering(G, weight="weight")
    vals = [cc[i] for i in range(len(cc))]
    return torch.tensor(vals, dtype=torch.float).unsqueeze(1)  # [84,1]


def compute_clustering_coefficient(matrix: pd.DataFrame) -> float:
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = float(matrix.iloc[u, v])
    return float(nx.average_clustering(G, weight="weight"))


def compute_path_length(matrix: pd.DataFrame) -> float:
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        w = float(matrix.iloc[u, v])
        d["distance"] = 1.0 / w if w > 0 else float("inf")

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    try:
        return float(nx.average_shortest_path_length(G, weight="distance"))
    except Exception:
        return float("nan")


# ============================================================
# Step 8: Threshold + log-transform connectomes
# ============================================================
def threshold_connectome(matrix: pd.DataFrame, percentile=95) -> pd.DataFrame:
    m = matrix.to_numpy()
    mask = ~np.eye(m.shape[0], dtype=bool)
    vals = m[mask]
    thr = np.percentile(vals, 100 - percentile)
    out = np.where(m >= thr, m, 0)
    return pd.DataFrame(out, index=matrix.index, columns=matrix.columns)


log_thresholded_connectomes = {}
for sid, mat in matched_connectomes.items():
    thr = threshold_connectome(mat, percentile=95)
    logm = np.log1p(thr)
    log_thresholded_connectomes[sid] = pd.DataFrame(logm, index=mat.index, columns=mat.columns)


# ============================================================
# Step 9: Matrix -> graph
# ============================================================
def matrix_to_graph(matrix: pd.DataFrame, device, sid: str, node_features_dict: dict):
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)

    node_feats = node_features_dict[sid]  # [84,3]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)  # [84,1]
    x = torch.cat([node_feats, clustering_tensor], dim=1)  # [84,4]
    x = 0.5 * x.to(device)

    return edge_index, edge_attr, x


# ============================================================
# Step 10: Add graph metrics to metadata (only subjects with PCA)
# ============================================================
addecode_metadata_pca = df_metadata_PCA_withConnectome.reset_index(drop=True).copy()
addecode_metadata_pca["MRI_Exam_fixed"] = addecode_metadata_pca["MRI_Exam_fixed"].astype(str).str.zfill(5)

addecode_metadata_pca["Clustering_Coeff"] = np.nan
addecode_metadata_pca["Path_Length"] = np.nan

for sid, mat in log_thresholded_connectomes.items():
    if sid not in set(addecode_metadata_pca["MRI_Exam_fixed"]):
        continue
    try:
        addecode_metadata_pca.loc[addecode_metadata_pca["MRI_Exam_fixed"] == sid, "Clustering_Coeff"] = compute_clustering_coefficient(mat)
        addecode_metadata_pca.loc[addecode_metadata_pca["MRI_Exam_fixed"] == sid, "Path_Length"] = compute_path_length(mat)
    except Exception as e:
        print(f"Failed metrics for {sid}: {e}")


# ============================================================
# Step 11: Encode + z-score metadata and PCA using healthy controls only
# ============================================================
le_sex = LabelEncoder()
addecode_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_metadata_pca["sex"].astype(str))

le_geno = LabelEncoder()
addecode_metadata_pca["genotype_enc"] = le_geno.fit_transform(addecode_metadata_pca["genotype"].astype(str))

numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ["PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"]
all_z_cols = numerical_cols + pca_cols

# coerce to numeric to avoid dtype surprises
for c in all_z_cols:
    addecode_metadata_pca[c] = pd.to_numeric(addecode_metadata_pca[c], errors="coerce")

df_controls = addecode_metadata_pca[~addecode_metadata_pca["Risk"].isin(["AD", "MCI"])].copy()
means = df_controls[all_z_cols].mean()
stds = df_controls[all_z_cols].std() + 1e-8

addecode_metadata_pca[all_z_cols] = (addecode_metadata_pca[all_z_cols] - means) / stds


# ============================================================
# Step 12: Build tensors
# ============================================================
subject_to_demographic_tensor = {}
subject_to_graphmetric_tensor = {}
subject_to_pca_tensor = {}

for _, row in addecode_metadata_pca.iterrows():
    sid = str(row["MRI_Exam_fixed"]).zfill(5)

    subject_to_demographic_tensor[sid] = torch.tensor(
        [
            float(row["Systolic"]),
            float(row["Diastolic"]),
            float(row["sex_encoded"]),
            float(row["genotype_enc"]),
        ],
        dtype=torch.float,
    )

    subject_to_graphmetric_tensor[sid] = torch.tensor(
        [float(row["Clustering_Coeff"]), float(row["Path_Length"])],
        dtype=torch.float,
    )

    subject_to_pca_tensor[sid] = torch.tensor(
        row[pca_cols].values.astype(np.float32),
        dtype=torch.float,
    )


# ============================================================
# Step 13: Build PyG graphs (only subjects that have ALL inputs)
# ============================================================
graph_data_list = []
kept_subjects = []

for sid, mat in log_thresholded_connectomes.items():
    try:
        if sid not in subject_to_demographic_tensor:
            continue
        if sid not in subject_to_graphmetric_tensor:
            continue
        if sid not in subject_to_pca_tensor:
            continue
        if sid not in normalized_node_features_dict:
            continue

        edge_index, edge_attr, x = matrix_to_graph(
            mat,
            device=torch.device("cpu"),
            sid=sid,
            node_features_dict=normalized_node_features_dict,
        )

        age_row = addecode_metadata_pca.loc[addecode_metadata_pca["MRI_Exam_fixed"] == sid, "age"]
        if age_row.empty:
            continue
        age = torch.tensor([float(age_row.values[0])], dtype=torch.float)

        global_feat = torch.cat(
            [
                subject_to_demographic_tensor[sid],   # 4
                subject_to_graphmetric_tensor[sid],   # 2
                subject_to_pca_tensor[sid],           # 10
            ],
            dim=0,
        )  # [16]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0),  # (1,16)
        )
        data.subject_id = sid

        graph_data_list.append(data)
        kept_subjects.append(sid)

    except Exception as e:
        print(f"Failed to process {sid}: {e}")

print(f"\nGraphs built: {len(graph_data_list)}")


# ============================================================
# Step 14: Load pretrained model + predict
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainAgeGATv2(torch.nn.Module):
    def __init__(self, global_feat_dim):
        super().__init__()
        import torch.nn as nn

        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
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
            nn.ReLU(),
        )

        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.pca_head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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
        return self.fc(x)


finalmodel_path = os.path.join(WORK, "ines/code/model_trained_on_all_healthy.pt")
print("\nLoading model from:", finalmodel_path)

model = BrainAgeGATv2(global_feat_dim=16).to(device)
state_dict = torch.load(finalmodel_path, map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.eval()

loader = DataLoader(graph_data_list, batch_size=1, shuffle=False)

subject_ids, true_ages, predicted_ages = [], [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        pred = float(model(batch).item())
        sid = batch.subject_id[0]  # already 5-digit string
        subject_ids.append(sid)
        true_ages.append(float(batch.y.item()))
        predicted_ages.append(pred)


# ============================================================
# Step 15: BAG + cBAG + align metadata
# ============================================================
df_preds = pd.DataFrame(
    {
        "MRI_Exam_fixed": subject_ids,     # <-- KEEP ONE ID COLUMN
        "Age": true_ages,
        "Predicted_Age": predicted_ages,
    }
)
df_preds["BAG"] = df_preds["Predicted_Age"] - df_preds["Age"]

reg = LinearRegression().fit(df_preds[["Age"]], df_preds["BAG"])
df_preds["cBAG"] = df_preds["BAG"] - reg.predict(df_preds[["Age"]])

y_true = np.array(true_ages)
y_pred = np.array(predicted_ages)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n===== MODEL PERFORMANCE (AD-DECODE) =====")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print("=========================================\n")

# align to metadata rows (PCA+connectome subset)
addecode_metadata_pca = addecode_metadata_pca.copy()
addecode_metadata_pca["MRI_Exam_fixed"] = addecode_metadata_pca["MRI_Exam_fixed"].astype(str).str.zfill(5)

df_preds_aligned = addecode_metadata_pca[["MRI_Exam_fixed"]].merge(
    df_preds, on="MRI_Exam_fixed", how="left"
)

meta_cols = [
    "Risk",
    "sex",
    "genotype",
    "APOE",
    "Systolic",
    "Diastolic",
    "BMI",
    "Clustering_Coeff",
    "Path_Length",
] + pca_cols

df_preds_aligned = df_preds_aligned.merge(
    addecode_metadata_pca[["MRI_Exam_fixed"] + meta_cols],
    on="MRI_Exam_fixed",
    how="left",
)

df_preds_aligned["Risk"] = df_preds_aligned["Risk"].fillna("NoRisk").replace(r"^\s*$", "NoRisk", regex=True)

out_csv = os.path.join(RESULTS_DIR, "brain_age_predictions_with_metadata.csv")
df_preds_aligned.to_csv(out_csv, index=False)
print("Saved:", out_csv)


# ============================================================
# Step 16: Read volume_norm (read-only) and attach ROI0 + hippocampus (17+53)
#   Specs:
#     - ROI 0 = TotalBrain (mL) in this file
#     - ROI 17 + ROI 53 = Hippocampus (%)
# ============================================================
print("\nVOLUME_NORM (read-only): ROI0 total brain (mL), ROI17+ROI53 hippocampus (%)\n")

# legend path relative to WORK (one read)
legend_path = os.path.join(WORK, "data/IITmean_RPI_label_index.csv")
if os.path.exists(legend_path):
    df_legend = pd.read_csv(legend_path)
    # best-effort detection
    cols_lower = {c: c.lower() for c in df_legend.columns}
    id_candidates = [c for c in df_legend.columns if any(k in cols_lower[c] for k in ["index", "id", "roi"])]
    name_candidates = [c for c in df_legend.columns if any(k in cols_lower[c] for k in ["name", "label", "region", "structure"])]
    if id_candidates and name_candidates:
        roi_id_col = id_candidates[0]
        roi_name_col = name_candidates[-1]
        df_legend[roi_id_col] = pd.to_numeric(df_legend[roi_id_col], errors="coerce")
        roi_to_name = dict(zip(df_legend[roi_id_col].dropna().astype(int), df_legend[roi_name_col].astype(str)))
        print("Legend lookup (best-effort):")
        print("ROI 0 :", roi_to_name.get(0, "Total brain (by your convention)"))
        print("ROI 17:", roi_to_name.get(17, "Left hippocampus"))
        print("ROI 53:", roi_to_name.get(53, "Right hippocampus"))
        print()
else:
    print("[WARN] Legend file not found at:", legend_path)
    print()

vol_path_norm = os.path.join(
    WORK,
    "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume_norm.txt",
)
df_vol_norm = pd.read_csv(vol_path_norm, sep="\t")
df_vol_norm = ensure_roi_column(df_vol_norm)

vol_subject_cols = [c for c in df_vol_norm.columns if str(c).startswith("S")]
df_vol_norm[vol_subject_cols] = df_vol_norm[vol_subject_cols].apply(pd.to_numeric, errors="coerce").astype(float)

# pull the rows we need
row0 = df_vol_norm.loc[df_vol_norm["ROI"] == 0, vol_subject_cols]
row17 = df_vol_norm.loc[df_vol_norm["ROI"] == 17, vol_subject_cols]
row53 = df_vol_norm.loc[df_vol_norm["ROI"] == 53, vol_subject_cols]

if row0.empty:
    raise ValueError("ROI 0 not found in volume_norm file.")
if row17.empty or row53.empty:
    raise ValueError("ROI 17 and/or ROI 53 not found in volume_norm file.")

total_ml_cols = row0.iloc[0]  # mL
hipp_pct_cols = row17.iloc[0] + row53.iloc[0]  # %
# if you ever want hippocampus mL from %, this is the conversion:
hipp_ml_cols = (hipp_pct_cols / 100.0) * total_ml_cols

# collapse multiple columns per subject id (master/replicas) by mean
col_sid = {c: col_to_sid(c) for c in vol_subject_cols}

def aggregate_by_sid(series_cols: pd.Series) -> pd.Series:
    tmp = {}
    for colname, val in series_cols.items():
        sid = col_sid[colname]
        tmp.setdefault(sid, []).append(val)
    return pd.Series({sid: float(np.nanmean(vals)) for sid, vals in tmp.items()})

total_ml_by_sid = aggregate_by_sid(total_ml_cols)
hipp_pct_by_sid = aggregate_by_sid(hipp_pct_cols)
hipp_ml_by_sid = aggregate_by_sid(hipp_ml_cols)

# attach to predictions (use ONE ID: MRI_Exam_fixed)
df_preds_aligned["MRI_Exam_fixed"] = df_preds_aligned["MRI_Exam_fixed"].astype(str).str.zfill(5)
df_preds_aligned["TotalBrain_mL"] = df_preds_aligned["MRI_Exam_fixed"].map(total_ml_by_sid)
df_preds_aligned["Hippocampus_pct"] = df_preds_aligned["MRI_Exam_fixed"].map(hipp_pct_by_sid)
df_preds_aligned["Hippocampus_mL"] = df_preds_aligned["MRI_Exam_fixed"].map(hipp_ml_by_sid)

print("Missing TotalBrain_mL:", int(df_preds_aligned["TotalBrain_mL"].isna().sum()))
print("Missing Hippocampus_pct:", int(df_preds_aligned["Hippocampus_pct"].isna().sum()))
print("Missing Hippocampus_mL:", int(df_preds_aligned["Hippocampus_mL"].isna().sum()))

out_csv2 = os.path.join(RESULTS_DIR, "brain_age_predictions_with_metadata_and_volumes.csv")
df_preds_aligned.to_csv(out_csv2, index=False)
print("Saved:", out_csv2)


# ============================================================
# Step 17: Plots (uniform) + save PNGs
# ============================================================
# cBAG vs Age
regplot_and_save(
    df=df_preds_aligned,
    x="Age",
    y="cBAG",
    title="cBAG vs Age",
    out_png=os.path.join(RESULTS_DIR, "cBAG_vs_Age.png"),
    add_hline0=True,
)

# cBAG vs Total Brain (mL)
regplot_and_save(
    df=df_preds_aligned,
    x="TotalBrain_mL",
    y="cBAG",
    title="cBAG vs Total Brain Volume (mL) [ROI 0]",
    out_png=os.path.join(RESULTS_DIR, "cBAG_vs_TotalBrain_mL.png"),
    add_hline0=True,
)

# cBAG vs Hippocampus (%)
regplot_and_save(
    df=df_preds_aligned,
    x="Hippocampus_pct",
    y="cBAG",
    title="cBAG vs Hippocampus (% of total) [ROI 17 + ROI 53]",
    out_png=os.path.join(RESULTS_DIR, "cBAG_vs_Hippocampus_pct.png"),
    add_hline0=True,
)

# Optional sanity plot: TotalBrain vs Hippocampus_mL
regplot_and_save(
    df=df_preds_aligned,
    x="TotalBrain_mL",
    y="Hippocampus_mL",
    title="Hippocampus (mL) vs Total Brain (mL)",
    out_png=os.path.join(RESULTS_DIR, "Hippocampus_mL_vs_TotalBrain_mL.png"),
    add_hline0=False,
)


# ============================================================
# Sanity checks: volumes vs Age
# ============================================================

# Hippocampus (%) vs Age
regplot_and_save(
    df=df_preds_aligned,
    x="Age",
    y="Hippocampus_pct",
    title="Hippocampus (% of total) vs Age [ROI 17 + ROI 53]",
    out_png=os.path.join(RESULTS_DIR, "Hippocampus_pct_vs_Age.png"),
    add_hline0=False,
)

# Total brain (mL) vs Age
regplot_and_save(
    df=df_preds_aligned,
    x="Age",
    y="TotalBrain_mL",
    title="Total Brain Volume (mL) vs Age [ROI 0]",
    out_png=os.path.join(RESULTS_DIR, "TotalBrain_mL_vs_Age.png"),
    add_hline0=False,
)


# ============================================================
# cBAG vs Hippocampus FA (ROI 17 + ROI 53)
# ============================================================

print("\nHippocampus FA (ROI 17 + ROI 53)\n")

# ---- Load FA file ----
fa_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt"
)

df_fa = pd.read_csv(fa_path, sep="\t")

# Ensure ROI column
if df_fa.columns[0] != "ROI":
    df_fa = df_fa.rename(columns={df_fa.columns[0]: "ROI"})

df_fa["ROI"] = pd.to_numeric(df_fa["ROI"], errors="coerce")

# Subject columns
fa_subject_cols = [c for c in df_fa.columns if str(c).startswith("S")]

# Convert numeric
df_fa[fa_subject_cols] = df_fa[fa_subject_cols].apply(pd.to_numeric, errors="coerce").astype(float)

# ---- Extract hippocampus ROIs ----
row17 = df_fa.loc[df_fa["ROI"] == 17, fa_subject_cols]
row53 = df_fa.loc[df_fa["ROI"] == 53, fa_subject_cols]

if row17.empty or row53.empty:
    raise ValueError("ROI 17 and/or 53 not found in FA file.")

# L + R mean FA
hipp_fa_cols = (row17.iloc[0] + row53.iloc[0]) / 2.0

# ---- Collapse columns to 5-digit subject IDs ----
def aggregate_by_sid(series_cols):
    tmp = {}
    for colname, val in series_cols.items():
        m = re.search(r"S(\d+)", str(colname))
        if not m:
            continue
        sid = m.group(1).zfill(5)
        tmp.setdefault(sid, []).append(val)
    return pd.Series({sid: float(np.nanmean(vals)) for sid, vals in tmp.items()})

hipp_fa_by_sid = aggregate_by_sid(hipp_fa_cols)

# ---- Attach to predictions ----
df_preds_aligned["MRI_Exam_fixed"] = df_preds_aligned["MRI_Exam_fixed"].astype(str).str.zfill(5)

df_preds_aligned["Hippocampus_FA"] = df_preds_aligned["MRI_Exam_fixed"].map(hipp_fa_by_sid)

print("Missing Hippocampus_FA:", df_preds_aligned["Hippocampus_FA"].isna().sum())

# ---- Correlation ----
df_plot = df_preds_aligned.dropna(subset=["cBAG", "Hippocampus_FA"]).copy()

r, p = pearsonr(df_plot["Hippocampus_FA"], df_plot["cBAG"])

print(f"\ncBAG vs Hippocampus FA: r={r:.3f}, p={p:.4g}, n={len(df_plot)}\n")

# ---- Plot ----
plt.figure(figsize=(7,6))

sns.regplot(
    data=df_plot,
    x="Hippocampus_FA",
    y="cBAG",
    ci=95,
    scatter_kws={"alpha":0.7},
    line_kws={"color":"darkred"}
)

plt.axhline(0, linestyle="--", color="gray")

plt.xlabel("Hippocampus FA (mean L+R)")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.title(f"cBAG vs Hippocampus FA\nr = {r:.3f}, p = {p:.4g}, n = {len(df_plot)}")

plt.tight_layout()

save_path = os.path.join(os.environ["WORK"], "ines/results/cBAG_vs_Hippocampus_FA.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved:", save_path)

# ============================================================
# cBAG vs Hippocampus MD (ROI 17 + ROI 53)
# ============================================================

print("\nHippocampus MD (ROI 17 + ROI 53)\n")

# ---- Load MD file ----
md_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/AD_DECODE/metadata/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt"
)

df_md = pd.read_csv(md_path, sep="\t")

# Ensure ROI column
if df_md.columns[0] != "ROI":
    df_md = df_md.rename(columns={df_md.columns[0]: "ROI"})

df_md["ROI"] = pd.to_numeric(df_md["ROI"], errors="coerce")

# Subject columns
md_subject_cols = [c for c in df_md.columns if str(c).startswith("S")]

# Convert numeric
df_md[md_subject_cols] = df_md[md_subject_cols].apply(pd.to_numeric, errors="coerce").astype(float)

# ---- Extract hippocampus ROIs ----
row17 = df_md.loc[df_md["ROI"] == 17, md_subject_cols]
row53 = df_md.loc[df_md["ROI"] == 53, md_subject_cols]

if row17.empty or row53.empty:
    raise ValueError("ROI 17 and/or 53 not found in MD file.")

# Mean MD L+R
hipp_md_cols = (row17.iloc[0] + row53.iloc[0]) / 2.0

# ---- Collapse to 5-digit subject IDs ----
def aggregate_by_sid(series_cols):
    tmp = {}
    for colname, val in series_cols.items():
        m = re.search(r"S(\d+)", str(colname))
        if not m:
            continue
        sid = m.group(1).zfill(5)
        tmp.setdefault(sid, []).append(val)
    return pd.Series({sid: float(np.nanmean(vals)) for sid, vals in tmp.items()})

hipp_md_by_sid = aggregate_by_sid(hipp_md_cols)

# ---- Attach to predictions ----
df_preds_aligned["MRI_Exam_fixed"] = df_preds_aligned["MRI_Exam_fixed"].astype(str).str.zfill(5)
df_preds_aligned["Hippocampus_MD"] = df_preds_aligned["MRI_Exam_fixed"].map(hipp_md_by_sid)

print("Missing Hippocampus_MD:", df_preds_aligned["Hippocampus_MD"].isna().sum())

# ---- Correlation ----
df_plot = df_preds_aligned.dropna(subset=["cBAG", "Hippocampus_MD"]).copy()

r, p = pearsonr(df_plot["Hippocampus_MD"], df_plot["cBAG"])

print(f"\ncBAG vs Hippocampus MD: r={r:.3f}, p={p:.4g}, n={len(df_plot)}\n")

# ---- Plot ----
plt.figure(figsize=(7,6))

sns.regplot(
    data=df_plot,
    x="Hippocampus_MD",
    y="cBAG",
    ci=95,
    scatter_kws={"alpha":0.7},
    line_kws={"color":"darkblue"}
)

plt.axhline(0, linestyle="--", color="gray")

plt.xlabel("Hippocampus MD (mean L+R)")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.title(f"cBAG vs Hippocampus MD\nr = {r:.3f}, p = {p:.4g}, n = {len(df_plot)}")

plt.tight_layout()

save_path = os.path.join(os.environ["WORK"], "ines/results/cBAG_vs_Hippocampus_MD.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved:", save_path)
print("\nDone.")