#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized edge-SHAP for harmonized cohorts
============================================

Works with:
    - ADNI
    - ADRC
    - HABS
    - AD_DECODE

What it does
------------
1) Loads prebuilt graph_data_list_<cohort>.pt graphs
2) Loads the harmonized training checkpoint
3) Verifies graph dimensions against checkpoint dimensions
4) Runs edge-SHAP with GradientExplainer over edge_attr
5) Saves one CSV per subject:
       edge_shap_subject_<SUBJECT>.csv
   and summary outputs

Important
---------
This version uses the SAME saved graphs that the harmonized training used.
That avoids all the dimension mismatch problems from rebuilding graphs by hand.
"""

import os
import re
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool
import shap

warnings.filterwarnings("ignore")

WORK = os.environ["WORK"]

# =========================================================
# USER CONFIG
# =========================================================

COHORT = "ADNI"   # "ADNI", "ADRC", "HABS", "AD_DECODE"
SEED = 42
TOP_N_EDGES = 20
MAX_SUBJECTS = None   # set to an int like 20 for quick testing, or None for all

# =========================================================
# PATHS
# =========================================================

COHORT_CONFIG = {
    "ADNI": {
        "graph_path": os.path.join(WORK, "ines/results/harmonized/ADNI/graphs/graph_data_list_adni.pt"),
        "model_path": os.path.join(WORK, "ines/results/BrainAgePredictionADNI/brainage_adni_prediction_model.pt"),
        "out_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_adni"),
    },
    "ADRC": {
        "graph_path": os.path.join(WORK, "ines/results/harmonized/ADRC/graphs/graph_data_list_adrc.pt"),
        "model_path": os.path.join(WORK, "ines/results/BrainAgePredictionADRC/brainage_adrc_prediction_model.pt"),
        "out_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_adrc"),
    },
    "HABS": {
        "graph_path": os.path.join(WORK, "ines/results/harmonized/HABS/graphs/graph_data_list_habs.pt"),
        "model_path": os.path.join(WORK, "ines/results/BrainAgePredictionHABS/brainage_habs_prediction_model.pt"),
        "out_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_habs"),
    },
    "AD_DECODE": {
        "graph_path": os.path.join(WORK, "ines/results/harmonized/AD_DECODE/graphs/graph_data_list_ad_decode.pt"),
        "model_path": os.path.join(WORK, "ines/results/BrainAgePredictionADDECODE/brainage_addecode_prediction_model.pt"),
        "out_dir": os.path.join(WORK, "ines/results/Shap_edges/edges_addecode"),
    },
}

if COHORT not in COHORT_CONFIG:
    raise ValueError(f"Invalid COHORT: {COHORT}. Choose from {list(COHORT_CONFIG.keys())}")

CFG = COHORT_CONFIG[COHORT]
GRAPH_PATH = CFG["graph_path"]
MODEL_PATH = CFG["model_path"]
EDGE_SHAP_DIR = CFG["out_dir"]
os.makedirs(EDGE_SHAP_DIR, exist_ok=True)

print("Cohort:", COHORT)
print("Graph path:", GRAPH_PATH)
print("Model path:", MODEL_PATH)
print("Output dir:", EDGE_SHAP_DIR)

# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =========================================================
# HELPERS
# =========================================================

def get_graph_identifier(data, idx):
    candidate_keys = [
        "subject_id", "graph_id", "connectome_key", "match_id",
        "PTID", "ptid", "regional_id"
    ]
    for key in candidate_keys:
        if hasattr(data, key):
            value = getattr(data, key)
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    continue
            if value is not None:
                return str(value)
    return f"graph_{idx}"

def prepare_graph_for_model(data):
    d = data.clone()

    if not hasattr(d, "edge_attr") or d.edge_attr is None:
        raise ValueError("Graph is missing edge_attr.")
    if d.edge_attr.dim() == 1:
        d.edge_attr = d.edge_attr.unsqueeze(-1)
    d.edge_attr = d.edge_attr.float()

    if not hasattr(d, "x") or d.x is None:
        raise ValueError("Graph is missing x.")
    d.x = d.x.float()

    if not hasattr(d, "global_features") or d.global_features is None:
        d.global_features = torch.zeros((1, 0), dtype=torch.float)
    else:
        if not torch.is_tensor(d.global_features):
            d.global_features = torch.tensor(d.global_features, dtype=torch.float)
        d.global_features = d.global_features.float()
        if d.global_features.dim() == 1:
            d.global_features = d.global_features.unsqueeze(0)

    if hasattr(d, "y") and d.y is not None:
        if not torch.is_tensor(d.y):
            d.y = torch.tensor([float(d.y)], dtype=torch.float)
        else:
            d.y = d.y.view(-1).float()

    return d

# =========================================================
# MODEL FROM HARMONIZED TRAINING
# =========================================================

class GNNBrainAge(nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64, dropout=0.2, edge_dim=1):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        self.conv3 = GINEConv(nn3, edge_dim=edge_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        fusion_in = hidden_dim + global_feat_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        gnn_emb = global_mean_pool(x, batch)

        if hasattr(data, "global_features") and data.global_features is not None:
            gf = data.global_features.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
        else:
            gf = torch.zeros((gnn_emb.shape[0], 0), device=gnn_emb.device)

        fused = torch.cat([gnn_emb, gf], dim=1)
        out = self.regressor(fused).squeeze(-1)
        return out

# =========================================================
# EDGE SHAP WRAPPER
# =========================================================

class EdgeSHAPWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, base_data):
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
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            d.edge_attr = ea.to(d.x.device)
            out = self.model(d)
            out = out.view(1, 1)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

# =========================================================
# LOAD GRAPHS
# =========================================================

if not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

graphs = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
graphs = [prepare_graph_for_model(g) for g in graphs]

if MAX_SUBJECTS is not None:
    graphs = graphs[:MAX_SUBJECTS]

print("Loaded graphs:", len(graphs))
if len(graphs) == 0:
    raise RuntimeError("No graphs loaded.")

g0 = graphs[0]
print("Example graph dims:")
print("  x shape         :", g0.x.shape)
print("  edge_attr shape :", g0.edge_attr.shape)
print("  global_features :", g0.global_features.shape)
print("  y shape         :", None if not hasattr(g0, "y") or g0.y is None else g0.y.shape)
print("  subject_id      :", getattr(g0, "subject_id", None))
print("  graph_id        :", getattr(g0, "graph_id", None))
print("  connectome_key  :", getattr(g0, "connectome_key", None))

# =========================================================
# LOAD CHECKPOINT
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=device)

if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
    raise ValueError(
        f"Checkpoint format not recognized for {MODEL_PATH}. "
        f"Expected dict with 'model_state_dict'."
    )

print("Checkpoint keys:", list(ckpt.keys()))
print("Checkpoint dims:")
print("  node_feat_dim  :", ckpt["node_feat_dim"])
print("  global_feat_dim:", ckpt["global_feat_dim"])
print("  edge_dim       :", ckpt["edge_dim"])
print("  hidden_dim     :", ckpt["hidden_dim"])
print("  dropout        :", ckpt["dropout"])

if g0.x.shape[1] != ckpt["node_feat_dim"]:
    raise ValueError(
        f"Graph node_feat_dim={g0.x.shape[1]} but checkpoint expects {ckpt['node_feat_dim']}"
    )
if g0.edge_attr.shape[1] != ckpt["edge_dim"]:
    raise ValueError(
        f"Graph edge_dim={g0.edge_attr.shape[1]} but checkpoint expects {ckpt['edge_dim']}"
    )
if g0.global_features.shape[1] != ckpt["global_feat_dim"]:
    raise ValueError(
        f"Graph global_feat_dim={g0.global_features.shape[1]} but checkpoint expects {ckpt['global_feat_dim']}"
    )

model = GNNBrainAge(
    node_feat_dim=ckpt["node_feat_dim"],
    global_feat_dim=ckpt["global_feat_dim"],
    hidden_dim=ckpt["hidden_dim"],
    dropout=ckpt["dropout"],
    edge_dim=ckpt["edge_dim"],
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Loaded model from:", MODEL_PATH)

# =========================================================
# RUN EDGE SHAP
# =========================================================

all_subject_summaries = []

for idx, data in enumerate(graphs, 1):
    sid = get_graph_identifier(data, idx)
    print(f"[{idx}/{len(graphs)}] Subject {sid}")

    try:
        base_data = data.clone().to(device)
        if base_data.edge_attr.dim() == 1:
            base_data.edge_attr = base_data.edge_attr.unsqueeze(-1)

        if not hasattr(base_data, "batch") or base_data.batch is None:
            base_data.batch = torch.zeros(base_data.num_nodes, dtype=torch.long, device=device)

        wrapper = EdgeSHAPWrapper(model, base_data)
        num_edges = base_data.edge_attr.shape[0]

        baseline = torch.zeros((1, num_edges, 1), dtype=torch.float32, device=device)
        input_ea = base_data.edge_attr.unsqueeze(0)

        explainer = shap.GradientExplainer(wrapper, baseline)
        shap_vals = explainer.shap_values(input_ea)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals)

        if shap_vals.ndim == 3:
            shap_edge = shap_vals[0, :, 0]
        elif shap_vals.ndim == 2:
            shap_edge = shap_vals[0, :]
        else:
            shap_edge = np.squeeze(shap_vals)

        edges = base_data.edge_index.detach().cpu().numpy().T

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

        print("    Saved:", out_csv)

    except Exception as e:
        print(f"    Failed for subject {sid}: {e}")

# =========================================================
# SAVE SUMMARY
# =========================================================

if len(all_subject_summaries) > 0:
    df_summary = pd.DataFrame(all_subject_summaries)
    summary_csv = os.path.join(EDGE_SHAP_DIR, "edge_shap_summary_all_subjects.csv")
    df_summary.to_csv(summary_csv, index=False)
    print("Saved summary:", summary_csv)

all_edge_tables = []
for fname in os.listdir(EDGE_SHAP_DIR):
    if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
        fpath = os.path.join(EDGE_SHAP_DIR, fname)
        df_tmp = pd.read_csv(fpath)
        df_tmp["edge_key"] = df_tmp["Node_i"].astype(str) + "_" + df_tmp["Node_j"].astype(str)
        all_edge_tables.append(df_tmp[["edge_key", "Node_i", "Node_j", "abs_SHAP", "SHAP_val"]])

if len(all_edge_tables) > 0:
    df_all_edges = pd.concat(all_edge_tables, ignore_index=True)

    df_mean_edges = (
        df_all_edges
        .groupby(["edge_key", "Node_i", "Node_j"], as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            mean_SHAP=("SHAP_val", "mean")
        )
        .sort_values("mean_abs_SHAP", ascending=False)
    )

    mean_csv = os.path.join(EDGE_SHAP_DIR, "edge_shap_mean_abs_all_subjects.csv")
    df_mean_edges.to_csv(mean_csv, index=False)
    print("Saved mean edge importance:", mean_csv)

    df_top = df_mean_edges.head(TOP_N_EDGES).copy().iloc[::-1]
    df_top["Edge"] = df_top["Node_i"].astype(str) + "-" + df_top["Node_j"].astype(str)

    plt.figure(figsize=(10, 7))
    plt.barh(df_top["Edge"], df_top["mean_abs_SHAP"])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"{COHORT}: Top {TOP_N_EDGES} most important edges across subjects")
    plt.tight_layout()
    top_fig = os.path.join(EDGE_SHAP_DIR, "edge_shap_top20_mean_abs.png")
    plt.savefig(top_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", top_fig)

print("DONE.")