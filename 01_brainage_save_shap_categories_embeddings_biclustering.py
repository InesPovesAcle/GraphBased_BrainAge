#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP computation + visualization for harmonized brain-age GNN ablation models
============================================================================

Adapted for the current output structure:

Graph builder outputs:
    $WORK/ines/results/harmonized/<COHORT>/graphs/<FEATURE_SET>/

Training outputs:
    $WORK/ines/results/BrainAgePrediction<COHORT>/ablation_<FEATURE_SET>/

This script loops over all cohorts and all feature-set ablations, loads the
matching graph/model/encoding files, computes SHAP values, saves per-subject
CSV files, population summaries, and publication-style figures.

Default behaviour
-----------------
By default, this script computes GLOBAL SHAP only, because it is the fastest and
most interpretable for the ablation models. You can enable node and edge SHAP in
USER CONFIG below.

Outputs
-------
For each cohort + feature_set:
    $WORK/ines/results/Shap_inputs_Ablations/<cohort_slug>/<feature_set>/

Combined summary:
    $WORK/ines/results/Shap_inputs_Ablations/combined_shap_run_summary.csv

Notes
-----
SHAP can only explain variables that are actual model inputs. Since each graph
file was already built for one feature_set, global_features already contains
only the inputs for that ablation. This script does not re-mask features.
"""

import os
import re
import json
import random
import traceback
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("error")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool
from joblib import Parallel, delayed
import shap

warnings.filterwarnings("ignore")

# =========================================================
# USER CONFIG
# =========================================================
WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")

COHORTS_TO_RUN = [
    "ADNI",
    "ADRC",
    "HABS",
    "AD_DECODE",
]

FEATURE_SETS_TO_RUN = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

# Use graph_data_list_<cohort>_all.pt for SHAP on the full cohort.
# Set False to explain only the healthy-control training set.
USE_ALL_SUBJECT_GRAPHS = True

SEED = 42
MAX_SUBJECTS = None       # e.g. 20 for testing, None for all available graphs
N_JOBS = 4                # SHAP is CPU-heavy. Start with 2-4 if unsure.
FORCE_CPU = True          # safer with joblib and SHAP; set False if you know GPU memory is OK

# Global SHAP is most useful for comparing ablation feature sets.
RUN_GLOBAL_FEATURE_SHAP = True
RUN_NODE_FEATURE_SHAP = True   # set False for a fast global-only test
RUN_EDGE_SHAP = False

TOP_N_EDGES = 20
TOP_N_GLOBAL_FEATURES = 20
TOP_N_NODE_FEATURES = 20
TOP_N_REGIONS = 20
MAKE_COMPOSITE_FIGURES = True
SKIP_IF_SUBJECT_FILES_EXIST = True

# =========================================================
# SUBJECT-LEVEL HIERARCHICAL HEATMAPS
# =========================================================
# After SHAP finishes, save subject x feature heatmaps with hierarchical clustering.
# Rows = subjects, columns = SHAP variables / embedding dimensions.
MAKE_GLOBAL_SHAP_CLUSTER_HEATMAP = True
MAKE_NODE_SHAP_CLUSTER_HEATMAP = True
MAKE_EDGE_SHAP_CLUSTER_HEATMAP = False
MAKE_EMBEDDING_CLUSTER_HEATMAP = True

# SHAP heatmap settings.
HEATMAP_VALUE_COLUMN = "SHAP_val"       # "SHAP_val" for signed effects, or "abs_SHAP" for magnitude
HEATMAP_STANDARDIZE_COLUMNS = True      # z-score each column across subjects for visualization
HEATMAP_TOP_N_GLOBAL_FEATURES = None    # None = all global features
HEATMAP_TOP_N_NODE_FEATURES = 50        # keep plot readable if node SHAP is enabled
HEATMAP_TOP_N_EDGE_FEATURES = 50        # keep plot readable if edge SHAP is enabled
HEATMAP_HIDE_SUBJECT_LABELS_IF_GT = 80

# Embedding heatmap settings.
# Options:
#   "gnn"              = pooled GNN graph embedding before global-feature fusion; best default
#   "fused"            = GNN graph embedding concatenated with global features
#   "regressor_hidden" = hidden representation immediately before final output layer
EMBEDDING_REPRESENTATION = "gnn"
EMBEDDING_STANDARDIZE_COLUMNS = True

# Publication-oriented heatmap layout.
HEATMAP_TITLE_FONTSIZE = 18
HEATMAP_AXIS_LABEL_FONTSIZE = 14
HEATMAP_XTICK_FONTSIZE = 9
HEATMAP_YTICK_FONTSIZE = 7
HEATMAP_CBAR_FONTSIZE = 12
HEATMAP_DPI = 400
GLOBAL_SHAP_HEATMAP_FIGSIZE = (13, 18)
CATEGORY_SHAP_HEATMAP_FIGSIZE = (11, 18)
NODE_SHAP_HEATMAP_FIGSIZE = (18, 18)
EDGE_SHAP_HEATMAP_FIGSIZE = (20, 18)
EMBEDDING_HEATMAP_FIGSIZE = (22, 18)

# Category-specific global SHAP heatmaps.
# These split global_features into interpretable groups.
MAKE_GLOBAL_CATEGORY_SHAP_HEATMAPS = True
GLOBAL_SHAP_CATEGORIES_TO_SAVE = [
    "demographics",
    "graph",
    "biomarkers",
    "cardiovascular",
    "other",
]

LABELS_PATH = os.path.join(WORK, "ines/data/atlas/IITmean_RPI_index.xlsx")
OUT_BASE = os.path.join(WORK, "ines/results/Shap_inputs_Ablations")
os.makedirs(OUT_BASE, exist_ok=True)

# =========================================================
# COHORT CONFIG
# =========================================================
COHORT_CONFIG = {
    "ADNI": {
        "cohort_dir": "ADNI",
        "cohort_slug": "adni",
        "file_prefix": "adni",
        "model_prefix": "adni",
        "prediction_dir": "BrainAgePredictionADNI",
    },
    "ADRC": {
        "cohort_dir": "ADRC",
        "cohort_slug": "adrc",
        "file_prefix": "adrc",
        "model_prefix": "adrc",
        "prediction_dir": "BrainAgePredictionADRC",
    },
    "HABS": {
        "cohort_dir": "HABS",
        "cohort_slug": "habs",
        "file_prefix": "habs",
        "model_prefix": "habs",
        "prediction_dir": "BrainAgePredictionHABS",
    },
    "AD_DECODE": {
        "cohort_dir": "AD_DECODE",
        "cohort_slug": "addecode",
        "file_prefix": "ad_decode",
        "model_prefix": "addecode",
        "prediction_dir": "BrainAgePredictionADDECODE",
    },
}

# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =========================================================
# GENERAL HELPERS
# =========================================================
def safe_filename(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def get_graph_identifier(data, idx):
    candidate_keys = [
        "subject_id", "graph_id", "connectome_key", "connectome_full_key",
        "match_id", "PTID", "ptid", "regional_id", "runno", "MRI_Exam"
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


def load_encoding_info(path):
    if path is None or not os.path.exists(path):
        print(f"Encoding info not found: {path}")
        return {}
    with open(path, "r") as f:
        info = json.load(f)
    return info


def _flatten_string_list(obj):
    if isinstance(obj, list):
        if all(not isinstance(x, (list, dict)) for x in obj):
            return [str(x) for x in obj]
    if isinstance(obj, dict):
        for subkey in [
            "names", "columns", "features", "feature_names", "cols",
            "selected_features", "input_features", "variables"
        ]:
            if subkey in obj:
                out = _flatten_string_list(obj[subkey])
                if out is not None:
                    return out
    return None


def find_feature_names(encoding_info, ckpt, key_candidates, expected_dim, fallback_prefix):
    # 1) Training checkpoint stores the exact selected global features.
    if fallback_prefix == "global_feature":
        for ckpt_key in ["selected_global_feature_names", "global_feature_names_all"]:
            if isinstance(ckpt, dict) and ckpt_key in ckpt:
                names = _flatten_string_list(ckpt[ckpt_key])
                if names is not None and len(names) == expected_dim:
                    return names

    # 2) Graph builder JSON stores global_feature_columns.
    for key in key_candidates:
        if key in encoding_info:
            names = _flatten_string_list(encoding_info[key])
            if names is not None and len(names) == expected_dim:
                return names

    # 3) Search nested dictionaries.
    for key, value in encoding_info.items():
        names = _flatten_string_list(value)
        if names is not None and len(names) == expected_dim:
            key_low = str(key).lower()
            if any(tok in key_low for tok in [fallback_prefix.split("_")[0], "global", "node", "edge", "feature"]):
                return names

    # 4) Useful node fallback for current graph builder.
    if fallback_prefix == "node_feature" and expected_dim == 3:
        return ["FA", "Volume", "Nodewise clustering"]

    return [f"{fallback_prefix}_{i}" for i in range(expected_dim)]


def load_node_labels(labels_path, node_ids):
    if not os.path.exists(labels_path):
        return {int(i): str(int(i)) for i in sorted(set(node_ids))}

    labels_df = pd.read_excel(labels_path)
    labels_df.columns = labels_df.columns.astype(str).str.strip()

    rename_map = {}
    for c in labels_df.columns:
        cl = c.lower()
        if cl in ["structure", "label", "region", "name", "roi", "roi_name"]:
            rename_map[c] = "Node_label"
        elif cl in ["index", "fs_index", "freesurfer_index"]:
            rename_map[c] = "fs_index"
        elif cl in ["index2", "node_index", "node", "node_id"]:
            rename_map[c] = "node_index"

    labels_df = labels_df.rename(columns=rename_map)
    if "Node_label" not in labels_df.columns:
        return {int(i): str(int(i)) for i in sorted(set(node_ids))}

    labels_df = labels_df.dropna(subset=["Node_label"]).copy()
    labels_df["Node_label"] = labels_df["Node_label"].astype(str).str.replace('"', '', regex=False).str.strip()

    node_ids = pd.Series(node_ids).dropna().astype(int)
    min_node = node_ids.min()
    max_node = node_ids.max()
    n_labels = len(labels_df)

    if min_node >= 0 and max_node < n_labels:
        if "node_index" in labels_df.columns:
            labels_ordered = labels_df.sort_values("node_index").reset_index(drop=True)
        else:
            labels_ordered = labels_df.reset_index(drop=True)
        return dict(zip(range(len(labels_ordered)), labels_ordered["Node_label"]))

    if "node_index" in labels_df.columns:
        tmp = labels_df.dropna(subset=["node_index"]).copy()
        tmp["node_index"] = tmp["node_index"].astype(int)
        if node_ids.isin(tmp["node_index"]).mean() > 0.8:
            return dict(zip(tmp["node_index"], tmp["Node_label"]))

    if "fs_index" in labels_df.columns:
        tmp = labels_df.dropna(subset=["fs_index"]).copy()
        tmp["fs_index"] = tmp["fs_index"].astype(int)
        if node_ids.isin(tmp["fs_index"]).mean() > 0.8:
            return dict(zip(tmp["fs_index"], tmp["Node_label"]))

    labels_ordered = labels_df.reset_index(drop=True)
    return dict(zip(range(len(labels_ordered)), labels_ordered["Node_label"]))


def get_input_paths(cohort, feature_set):
    cfg = COHORT_CONFIG[cohort]
    graph_dir = os.path.join(
        WORK,
        "ines/results/harmonized",
        cfg["cohort_dir"],
        "graphs",
        feature_set,
    )
    graph_name = (
        f"graph_data_list_{cfg['file_prefix']}_all.pt"
        if USE_ALL_SUBJECT_GRAPHS
        else f"graph_data_list_{cfg['file_prefix']}.pt"
    )
    metadata_name = (
        f"{cfg['file_prefix']}_metadata_all_aligned.csv"
        if USE_ALL_SUBJECT_GRAPHS
        else f"{cfg['file_prefix']}_metadata_aligned.csv"
    )
    metadata_raw_name = (
        f"{cfg['file_prefix']}_metadata_all_aligned_raw.csv"
        if USE_ALL_SUBJECT_GRAPHS
        else f"{cfg['file_prefix']}_metadata_aligned_raw.csv"
    )

    model_dir = os.path.join(
        WORK,
        "ines/results",
        cfg["prediction_dir"],
        f"ablation_{feature_set}",
    )
    model_path = os.path.join(
        model_dir,
        f"brainage_{cfg['model_prefix']}_{feature_set}_prediction_model.pt",
    )
    model_bc_path = os.path.join(
        model_dir,
        f"brainage_{cfg['model_prefix']}_{feature_set}_prediction_BIAS_CORRECTED_model.pt",
    )

    out_root = os.path.join(OUT_BASE, cfg["cohort_slug"], feature_set)

    return {
        "graph_path": os.path.join(graph_dir, graph_name),
        "metadata_path": os.path.join(graph_dir, metadata_name),
        "metadata_raw_path": os.path.join(graph_dir, metadata_raw_name),
        "encoding_info_path": os.path.join(graph_dir, f"{cfg['file_prefix']}_feature_encoding_info.json"),
        "model_path": model_path,
        "model_bc_path": model_bc_path,
        "model_dir": model_dir,
        "out_root": out_root,
    }

# =========================================================
# MODEL
# =========================================================
class GNNBrainAge(nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64, dropout=0.2, edge_dim=1):
        super().__init__()

        nn1 = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

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

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))

        gnn_emb = global_mean_pool(x, batch)

        if hasattr(data, "global_features") and data.global_features is not None:
            gf = data.global_features.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
        else:
            gf = torch.zeros((gnn_emb.shape[0], 0), device=gnn_emb.device)

        fused = torch.cat([gnn_emb, gf], dim=1)
        return self.regressor(fused).squeeze(-1)

# =========================================================
# SHAP WRAPPERS
# =========================================================
class EdgeSHAPWrapper(torch.nn.Module):
    def __init__(self, model, base_data):
        super().__init__()
        self.model = model
        self.base_data = base_data.clone().to(next(model.parameters()).device)
        if not hasattr(self.base_data, "batch") or self.base_data.batch is None:
            self.base_data.batch = torch.zeros(self.base_data.num_nodes, dtype=torch.long, device=self.base_data.x.device)

    def forward(self, edge_attr_batch):
        outputs = []
        for ea in edge_attr_batch:
            d = self.base_data.clone()
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            d.edge_attr = ea.to(d.x.device)
            outputs.append(self.model(d).view(1, 1))
        return torch.cat(outputs, dim=0)


class GlobalFeatureSHAPWrapper(torch.nn.Module):
    def __init__(self, model, base_data):
        super().__init__()
        self.model = model
        self.base_data = base_data.clone().to(next(model.parameters()).device)
        if not hasattr(self.base_data, "batch") or self.base_data.batch is None:
            self.base_data.batch = torch.zeros(self.base_data.num_nodes, dtype=torch.long, device=self.base_data.x.device)

    def forward(self, global_feature_batch):
        outputs = []
        for gf in global_feature_batch:
            d = self.base_data.clone()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            d.global_features = gf.to(d.x.device)
            outputs.append(self.model(d).view(1, 1))
        return torch.cat(outputs, dim=0)


class NodeFeatureSHAPWrapper(torch.nn.Module):
    def __init__(self, model, base_data):
        super().__init__()
        self.model = model
        self.base_data = base_data.clone().to(next(model.parameters()).device)
        if not hasattr(self.base_data, "batch") or self.base_data.batch is None:
            self.base_data.batch = torch.zeros(self.base_data.num_nodes, dtype=torch.long, device=self.base_data.x.device)

    def forward(self, node_feature_batch):
        outputs = []
        for x_new in node_feature_batch:
            d = self.base_data.clone()
            d.x = x_new.to(d.edge_attr.device)
            outputs.append(self.model(d).view(1, 1))
        return torch.cat(outputs, dim=0)

# =========================================================
# SHAP COMPUTATION
# =========================================================
def build_model_from_checkpoint(ckpt, device):
    model = GNNBrainAge(
        node_feat_dim=int(ckpt["node_feat_dim"]),
        global_feat_dim=int(ckpt["global_feat_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt["dropout"]),
        edge_dim=int(ckpt["edge_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_graphs_and_model(paths):
    missing = [k for k in ["graph_path", "model_path", "encoding_info_path"] if not os.path.exists(paths[k])]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join([f"{k}={paths[k]}" for k in missing]))

    graphs = torch.load(paths["graph_path"], map_location="cpu", weights_only=False)
    graphs = [prepare_graph_for_model(g) for g in graphs]
    if MAX_SUBJECTS is not None:
        graphs = graphs[:MAX_SUBJECTS]
    if len(graphs) == 0:
        raise RuntimeError("No graphs loaded.")

    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(paths["model_path"], map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint format not recognized: {paths['model_path']}")

    g0 = graphs[0]
    if g0.x.shape[1] != int(ckpt["node_feat_dim"]):
        raise ValueError(f"Graph node_feat_dim={g0.x.shape[1]} but checkpoint expects {ckpt['node_feat_dim']}")
    if g0.edge_attr.shape[1] != int(ckpt["edge_dim"]):
        raise ValueError(f"Graph edge_dim={g0.edge_attr.shape[1]} but checkpoint expects {ckpt['edge_dim']}")
    if g0.global_features.shape[1] != int(ckpt["global_feat_dim"]):
        raise ValueError(f"Graph global_feat_dim={g0.global_features.shape[1]} but checkpoint expects {ckpt['global_feat_dim']}")

    encoding_info = load_encoding_info(paths["encoding_info_path"])

    node_feature_names = find_feature_names(
        encoding_info,
        ckpt,
        key_candidates=["node_feature_names", "node_features", "node_cols", "regional_feature_names", "x_feature_names"],
        expected_dim=int(ckpt["node_feat_dim"]),
        fallback_prefix="node_feature",
    )
    global_feature_names = find_feature_names(
        encoding_info,
        ckpt,
        key_candidates=[
            "global_feature_columns", "global_feature_names", "global_features", "global_cols",
            "metadata_features", "graph_feature_names", "clinical_feature_names", "biomarker_features",
            "transcriptomic_features", "transcriptomics", "sex_features", "graph_features",
        ],
        expected_dim=int(ckpt["global_feat_dim"]),
        fallback_prefix="global_feature",
    )
    edge_feature_names = find_feature_names(
        encoding_info,
        ckpt,
        key_candidates=["edge_feature_names", "edge_features", "edge_cols", "connectivity_feature_names"],
        expected_dim=int(ckpt["edge_dim"]),
        fallback_prefix="edge_feature",
    )

    all_node_ids = []
    for g in graphs:
        all_node_ids.extend(g.edge_index.detach().cpu().numpy().flatten().tolist())
    node_label_map = load_node_labels(LABELS_PATH, all_node_ids)

    return graphs, ckpt, device, node_feature_names, global_feature_names, edge_feature_names, node_label_map


def compute_one_subject(
    idx,
    data,
    ckpt,
    node_feature_names,
    global_feature_names,
    edge_feature_names,
    node_label_map,
    edge_dir,
    node_dir,
    global_dir,
):
    sid = get_graph_identifier(data, idx)
    sid_safe = safe_filename(sid)

    subject_summary = {
        "Subject_ID": sid,
        "Subject_ID_safe": sid_safe,
        "Pred_Age": np.nan,
        "edge_status": "not_run",
        "node_status": "not_run",
        "global_status": "not_run",
        "edge_error": None,
        "node_error": None,
        "global_error": None,
    }

    edge_out = os.path.join(edge_dir, f"edge_shap_subject_{sid_safe}.csv")
    node_out = os.path.join(node_dir, f"node_feature_shap_subject_{sid_safe}.csv")
    global_out = os.path.join(global_dir, f"global_feature_shap_subject_{sid_safe}.csv")

    try:
        local_device = torch.device("cpu")
        local_model = build_model_from_checkpoint(ckpt, local_device)
        base_data = data.clone().to(local_device)

        if base_data.edge_attr.dim() == 1:
            base_data.edge_attr = base_data.edge_attr.unsqueeze(-1)
        if not hasattr(base_data, "batch") or base_data.batch is None:
            base_data.batch = torch.zeros(base_data.num_nodes, dtype=torch.long, device=local_device)

        with torch.no_grad():
            subject_summary["Pred_Age"] = float(local_model(base_data).detach().cpu().item())

        if RUN_GLOBAL_FEATURE_SHAP:
            if SKIP_IF_SUBJECT_FILES_EXIST and os.path.exists(global_out):
                subject_summary["global_status"] = "exists"
            else:
                try:
                    n_global = base_data.global_features.shape[1]
                    if n_global == 0:
                        subject_summary["global_status"] = "skipped"
                        subject_summary["global_error"] = "No global features"
                    else:
                        wrapper = GlobalFeatureSHAPWrapper(local_model, base_data)
                        baseline = torch.zeros((1, n_global), dtype=torch.float32, device=local_device)
                        input_gf = base_data.global_features.clone()
                        explainer = shap.GradientExplainer(wrapper, baseline)
                        shap_vals = explainer.shap_values(input_gf)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        shap_vals = np.array(shap_vals).squeeze()
                        if shap_vals.ndim == 0:
                            shap_vals = np.array([float(shap_vals)])
                        values = base_data.global_features.detach().cpu().numpy().squeeze()
                        if np.ndim(values) == 0:
                            values = np.array([float(values)])
                        df_global = pd.DataFrame({
                            "feature_index": np.arange(n_global),
                            "feature_name": global_feature_names,
                            "feature_value": values,
                            "SHAP_val": shap_vals,
                            "abs_SHAP": np.abs(shap_vals),
                        })
                        df_global.to_csv(global_out, index=False)
                        subject_summary["global_status"] = "ok"
                        subject_summary["MeanAbsSHAP_global"] = float(df_global["abs_SHAP"].mean())
                        subject_summary["MaxAbsSHAP_global"] = float(df_global["abs_SHAP"].max())
                except Exception as e:
                    subject_summary["global_status"] = "failed"
                    subject_summary["global_error"] = str(e)

        if RUN_NODE_FEATURE_SHAP:
            if SKIP_IF_SUBJECT_FILES_EXIST and os.path.exists(node_out):
                subject_summary["node_status"] = "exists"
            else:
                try:
                    wrapper = NodeFeatureSHAPWrapper(local_model, base_data)
                    n_nodes, n_node_features = base_data.x.shape
                    baseline = torch.zeros((1, n_nodes, n_node_features), dtype=torch.float32, device=local_device)
                    input_x = base_data.x.unsqueeze(0)
                    explainer = shap.GradientExplainer(wrapper, baseline)
                    shap_vals = explainer.shap_values(input_x)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    shap_vals = np.array(shap_vals).squeeze()
                    if shap_vals.ndim == 1:
                        shap_vals = shap_vals.reshape(n_nodes, n_node_features)
                    x_values = base_data.x.detach().cpu().numpy()
                    rows = []
                    for node_idx in range(n_nodes):
                        node_label = node_label_map.get(int(node_idx), str(node_idx))
                        for feat_idx in range(n_node_features):
                            rows.append({
                                "node_index": node_idx,
                                "node_label": node_label,
                                "feature_index": feat_idx,
                                "feature_name": node_feature_names[feat_idx],
                                "feature_value": float(x_values[node_idx, feat_idx]),
                                "SHAP_val": float(shap_vals[node_idx, feat_idx]),
                                "abs_SHAP": float(abs(shap_vals[node_idx, feat_idx])),
                            })
                    df_node = pd.DataFrame(rows)
                    df_node.to_csv(node_out, index=False)
                    subject_summary["node_status"] = "ok"
                    subject_summary["MeanAbsSHAP_node"] = float(df_node["abs_SHAP"].mean())
                    subject_summary["MaxAbsSHAP_node"] = float(df_node["abs_SHAP"].max())
                except Exception as e:
                    subject_summary["node_status"] = "failed"
                    subject_summary["node_error"] = str(e)

        if RUN_EDGE_SHAP:
            if SKIP_IF_SUBJECT_FILES_EXIST and os.path.exists(edge_out):
                subject_summary["edge_status"] = "exists"
            else:
                try:
                    wrapper = EdgeSHAPWrapper(local_model, base_data)
                    num_edges = base_data.edge_attr.shape[0]
                    edge_dim = base_data.edge_attr.shape[1]
                    baseline = torch.zeros((1, num_edges, edge_dim), dtype=torch.float32, device=local_device)
                    input_ea = base_data.edge_attr.unsqueeze(0)
                    explainer = shap.GradientExplainer(wrapper, baseline)
                    shap_vals = explainer.shap_values(input_ea)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    shap_edge = np.squeeze(np.array(shap_vals))
                    if shap_edge.ndim == 1:
                        shap_edge = shap_edge.reshape(num_edges, 1)
                    edges = base_data.edge_index.detach().cpu().numpy().T
                    edge_values = base_data.edge_attr.detach().cpu().numpy()
                    rows = []
                    for e_idx in range(num_edges):
                        ni, nj = int(edges[e_idx, 0]), int(edges[e_idx, 1])
                        si = node_label_map.get(ni, str(ni))
                        sj = node_label_map.get(nj, str(nj))
                        edge_label = f"{si} -- {sj}"
                        for feat_idx in range(edge_dim):
                            rows.append({
                                "edge_index": e_idx,
                                "Node_i": ni,
                                "Node_j": nj,
                                "Structure_i": si,
                                "Structure_j": sj,
                                "Edge": edge_label,
                                "edge_feature_index": feat_idx,
                                "edge_feature_name": edge_feature_names[feat_idx],
                                "edge_weight": float(edge_values[e_idx, feat_idx]),
                                "SHAP_val": float(shap_edge[e_idx, feat_idx]),
                                "abs_SHAP": float(abs(shap_edge[e_idx, feat_idx])),
                            })
                    df_edge = pd.DataFrame(rows)
                    df_edge.to_csv(edge_out, index=False)
                    subject_summary["edge_status"] = "ok"
                    subject_summary["MeanAbsSHAP_edge"] = float(df_edge["abs_SHAP"].mean())
                    subject_summary["MaxAbsSHAP_edge"] = float(df_edge["abs_SHAP"].max())
                except Exception as e:
                    subject_summary["edge_status"] = "failed"
                    subject_summary["edge_error"] = str(e)

    except Exception as e:
        err = str(e)
        subject_summary["edge_status"] = "failed" if RUN_EDGE_SHAP else "not_run"
        subject_summary["node_status"] = "failed" if RUN_NODE_FEATURE_SHAP else "not_run"
        subject_summary["global_status"] = "failed" if RUN_GLOBAL_FEATURE_SHAP else "not_run"
        subject_summary["edge_error"] = err
        subject_summary["node_error"] = err
        subject_summary["global_error"] = err

    return subject_summary

# =========================================================
# SUMMARY + PLOTS
# =========================================================
def summarize_edge_shap(edge_dir):
    tables = []
    if not os.path.isdir(edge_dir):
        return None
    for fname in os.listdir(edge_dir):
        if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(edge_dir, fname))
            df["subject"] = fname.replace("edge_shap_subject_", "").replace(".csv", "")
            tables.append(df)
    if len(tables) == 0:
        return None
    df_all = pd.concat(tables, ignore_index=True)
    group_cols = ["Node_i", "Node_j", "Structure_i", "Structure_j", "Edge", "edge_feature_index", "edge_feature_name"]
    summary = (
        df_all.groupby(group_cols, as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
    )
    summary.to_csv(os.path.join(edge_dir, "edge_shap_summary_all_subjects.csv"), index=False)
    return summary


def summarize_global_feature_shap(global_dir):
    tables = []
    if not os.path.isdir(global_dir):
        return None
    for fname in os.listdir(global_dir):
        if fname.startswith("global_feature_shap_subject_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(global_dir, fname))
            df["subject"] = fname.replace("global_feature_shap_subject_", "").replace(".csv", "")
            tables.append(df)
    if len(tables) == 0:
        return None
    df_all = pd.concat(tables, ignore_index=True)
    summary = (
        df_all.groupby(["feature_index", "feature_name"], as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            mean_feature_value=("feature_value", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
    )
    summary.to_csv(os.path.join(global_dir, "global_feature_shap_summary_all_subjects.csv"), index=False)
    return summary


def summarize_node_feature_shap(node_dir):
    tables = []
    if not os.path.isdir(node_dir):
        return None
    for fname in os.listdir(node_dir):
        if fname.startswith("node_feature_shap_subject_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(node_dir, fname))
            df["subject"] = fname.replace("node_feature_shap_subject_", "").replace(".csv", "")
            tables.append(df)
    if len(tables) == 0:
        return None
    df_all = pd.concat(tables, ignore_index=True)
    summary = (
        df_all.groupby(["node_index", "node_label", "feature_index", "feature_name"], as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            mean_feature_value=("feature_value", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
    )
    summary.to_csv(os.path.join(node_dir, "node_feature_shap_summary_all_subjects.csv"), index=False)
    return summary


def summarize_region_from_edge_shap(edge_summary, region_dir):
    if edge_summary is None or len(edge_summary) == 0:
        return None
    rows = []
    for _, r in edge_summary.iterrows():
        for node_col, label_col in [("Node_i", "Structure_i"), ("Node_j", "Structure_j")]:
            rows.append({
                "node_index": int(r[node_col]),
                "region": str(r[label_col]),
                "edge_feature_index": int(r["edge_feature_index"]),
                "edge_feature_name": str(r["edge_feature_name"]),
                "mean_abs_SHAP": float(r["mean_abs_SHAP"]),
                "mean_SHAP": float(r["mean_SHAP"]),
                "n_subjects": int(r["n_subjects"]),
            })
    region_long = pd.DataFrame(rows)
    region_summary = (
        region_long.groupby(["node_index", "region", "edge_feature_index", "edge_feature_name"], as_index=False)
        .agg(
            population_mean_abs_SHAP=("mean_abs_SHAP", "mean"),
            population_sum_abs_SHAP=("mean_abs_SHAP", "sum"),
            population_signed_mean_SHAP=("mean_SHAP", "mean"),
            n_edge_entries=("mean_abs_SHAP", "count"),
        )
        .sort_values("population_mean_abs_SHAP", ascending=False)
    )
    ensure_dir(region_dir)
    region_summary.to_csv(os.path.join(region_dir, "region_from_edge_shap_summary.csv"), index=False)
    return region_summary


def plot_top_bar(summary, label_col, value_col, title, xlabel, out_path, top_n=20):
    if summary is None or len(summary) == 0 or label_col not in summary.columns or value_col not in summary.columns:
        return False
    top = summary.sort_values(value_col, ascending=False).head(top_n).iloc[::-1].copy()
    plt.figure(figsize=(12, 8))
    plt.barh(top[label_col].astype(str), top[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_signed_bar(summary, label_col, signed_col, rank_col, title, xlabel, out_path, top_n=20):
    if summary is None or len(summary) == 0:
        return False
    top = summary.sort_values(rank_col, ascending=False).head(top_n).iloc[::-1].copy()
    colors = ["steelblue" if x > 0 else "crimson" for x in top[signed_col]]
    plt.figure(figsize=(12, 8))
    plt.barh(top[label_col].astype(str), top[signed_col], color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_composite_summary(summary, label_col, abs_col, signed_col, title, out_path, top_n=20):
    if summary is None or len(summary) == 0:
        return False
    top_abs = summary.sort_values(abs_col, ascending=False).head(top_n).iloc[::-1].copy()
    top_pos = summary.query(f"{signed_col} > 0").sort_values(signed_col, ascending=False).head(top_n).iloc[::-1].copy()
    top_neg = summary.query(f"{signed_col} < 0").sort_values(signed_col, ascending=True).head(top_n).iloc[::-1].copy()

    fig, axes = plt.subplots(1, 3, figsize=(26, 10))
    axes[0].barh(top_abs[label_col].astype(str), top_abs[abs_col])
    axes[0].set_title(f"A. Top {top_n} by mean |SHAP|")
    axes[0].set_xlabel("Mean |SHAP|")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    if len(top_pos) > 0:
        axes[1].barh(top_pos[label_col].astype(str), top_pos[signed_col], color="steelblue")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"B. Top {top_n} positive mean SHAP")
    axes[1].set_xlabel("Mean SHAP")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    if len(top_neg) > 0:
        axes[2].barh(top_neg[label_col].astype(str), top_neg[signed_col], color="crimson")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title(f"C. Top {top_n} negative mean SHAP")
    axes[2].set_xlabel("Mean SHAP")
    axes[2].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def save_feature_name_summary(out_root, node_feature_names, global_feature_names, edge_feature_names):
    feature_name_summary = pd.DataFrame({
        "category": ["node"] * len(node_feature_names) + ["global"] * len(global_feature_names) + ["edge"] * len(edge_feature_names),
        "feature_index": list(range(len(node_feature_names))) + list(range(len(global_feature_names))) + list(range(len(edge_feature_names))),
        "feature_name": node_feature_names + global_feature_names + edge_feature_names,
    })
    feature_name_summary.to_csv(os.path.join(out_root, "detected_model_input_feature_names.csv"), index=False)


def make_plots(cohort, feature_set, out_root, edge_summary, global_summary, node_summary, region_summary):
    plots_dir = ensure_dir(os.path.join(out_root, "plots"))
    composite_dir = ensure_dir(os.path.join(plots_dir, "composite"))
    plot_rows = []

    tag = f"{cohort}_{feature_set}"

    if edge_summary is not None:
        edge_summary = edge_summary.copy()
        edge_summary["label"] = edge_summary["Edge"].astype(str)
        if "edge_feature_name" in edge_summary.columns and edge_summary["edge_feature_name"].nunique() > 1:
            edge_summary["label"] = edge_summary["Edge"].astype(str) + " | " + edge_summary["edge_feature_name"].astype(str)

        out = os.path.join(plots_dir, f"{tag}_top_edges_mean_abs_SHAP.png")
        ok = plot_top_bar(edge_summary, "label", "mean_abs_SHAP", f"{cohort} {feature_set}: Top edges by mean |SHAP|", "Mean |SHAP|", out, TOP_N_EDGES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        out = os.path.join(plots_dir, f"{tag}_top_edges_signed_mean_SHAP.png")
        ok = plot_signed_bar(edge_summary, "label", "mean_SHAP", "mean_abs_SHAP", f"{cohort} {feature_set}: Top edges by signed mean SHAP", "Mean SHAP", out, TOP_N_EDGES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        if MAKE_COMPOSITE_FIGURES:
            out = os.path.join(composite_dir, f"{tag}_composite_edge_SHAP.png")
            ok = plot_composite_summary(edge_summary, "label", "mean_abs_SHAP", "mean_SHAP", f"{cohort} {feature_set}: Composite edge SHAP summary", out, TOP_N_EDGES)
            plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

    if global_summary is not None:
        out = os.path.join(plots_dir, f"{tag}_top_global_features_mean_abs_SHAP.png")
        ok = plot_top_bar(global_summary, "feature_name", "mean_abs_SHAP", f"{cohort} {feature_set}: Top global inputs by mean |SHAP|", "Mean |SHAP|", out, TOP_N_GLOBAL_FEATURES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        out = os.path.join(plots_dir, f"{tag}_top_global_features_signed_mean_SHAP.png")
        ok = plot_signed_bar(global_summary, "feature_name", "mean_SHAP", "mean_abs_SHAP", f"{cohort} {feature_set}: Top global inputs by signed mean SHAP", "Mean SHAP", out, TOP_N_GLOBAL_FEATURES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        if MAKE_COMPOSITE_FIGURES:
            out = os.path.join(composite_dir, f"{tag}_composite_global_feature_SHAP.png")
            ok = plot_composite_summary(global_summary, "feature_name", "mean_abs_SHAP", "mean_SHAP", f"{cohort} {feature_set}: Composite global input SHAP summary", out, TOP_N_GLOBAL_FEATURES)
            plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

    if node_summary is not None:
        node_summary = node_summary.copy()
        node_summary["label"] = node_summary["node_label"].astype(str) + " | " + node_summary["feature_name"].astype(str)

        out = os.path.join(plots_dir, f"{tag}_top_node_features_mean_abs_SHAP.png")
        ok = plot_top_bar(node_summary, "label", "mean_abs_SHAP", f"{cohort} {feature_set}: Top node-feature inputs by mean |SHAP|", "Mean |SHAP|", out, TOP_N_NODE_FEATURES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        out = os.path.join(plots_dir, f"{tag}_top_node_features_signed_mean_SHAP.png")
        ok = plot_signed_bar(node_summary, "label", "mean_SHAP", "mean_abs_SHAP", f"{cohort} {feature_set}: Top node-feature inputs by signed mean SHAP", "Mean SHAP", out, TOP_N_NODE_FEATURES)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        if MAKE_COMPOSITE_FIGURES:
            out = os.path.join(composite_dir, f"{tag}_composite_node_feature_SHAP.png")
            ok = plot_composite_summary(node_summary, "label", "mean_abs_SHAP", "mean_SHAP", f"{cohort} {feature_set}: Composite node-feature SHAP summary", out, TOP_N_NODE_FEATURES)
            plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

    if region_summary is not None:
        region_summary = region_summary.copy()
        region_summary["label"] = region_summary["region"].astype(str)
        if "edge_feature_name" in region_summary.columns and region_summary["edge_feature_name"].nunique() > 1:
            region_summary["label"] = region_summary["region"].astype(str) + " | " + region_summary["edge_feature_name"].astype(str)

        out = os.path.join(plots_dir, f"{tag}_top_regions_from_edge_mean_abs_SHAP.png")
        ok = plot_top_bar(region_summary, "label", "population_mean_abs_SHAP", f"{cohort} {feature_set}: Top regions from edge SHAP", "Population mean incident |SHAP|", out, TOP_N_REGIONS)
        plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

        if MAKE_COMPOSITE_FIGURES:
            out = os.path.join(composite_dir, f"{tag}_composite_region_from_edge_SHAP.png")
            ok = plot_composite_summary(region_summary, "label", "population_mean_abs_SHAP", "population_signed_mean_SHAP", f"{cohort} {feature_set}: Composite region SHAP from edge SHAP", out, TOP_N_REGIONS)
            plot_rows.append({"plot": out, "status": "saved" if ok else "skipped"})

    pd.DataFrame(plot_rows).to_csv(os.path.join(out_root, "shap_plot_log.csv"), index=False)
    return plot_rows



# =========================================================
# SUBJECT-LEVEL HIERARCHICAL HEATMAPS
# =========================================================
def _load_subject_level_long_table(input_dir, file_prefix, label_col):
    """Load all per-subject SHAP CSV files into one long table."""
    rows = []
    if not os.path.exists(input_dir):
        return None

    for fname in sorted(os.listdir(input_dir)):
        if not (fname.startswith(file_prefix) and fname.endswith(".csv")):
            continue
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if df.empty or label_col not in df.columns:
            continue
        subject_id = fname.replace(file_prefix, "").replace(".csv", "")
        df = df.copy()
        df["subject"] = subject_id
        rows.append(df)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _standardize_columns(df):
    """Z-score each feature/embedding dimension across subjects."""
    out = df.copy().astype(float)
    means = out.mean(axis=0)
    stds = out.std(axis=0, ddof=0).replace(0, np.nan)
    out = (out - means) / stds
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_subject_feature_matrix(long_df, label_col, value_col, top_n=None, ranking_col="abs_SHAP"):
    """Convert subject-level long SHAP table to subject x feature matrix."""
    if long_df is None or long_df.empty:
        return None
    if label_col not in long_df.columns or value_col not in long_df.columns:
        return None

    use_df = long_df.copy()
    use_df[label_col] = use_df[label_col].astype(str)

    if top_n is not None:
        if ranking_col in use_df.columns:
            keep = (
                use_df.groupby(label_col)[ranking_col]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index
            )
        else:
            keep = use_df[label_col].value_counts().head(top_n).index
        use_df = use_df[use_df[label_col].isin(keep)].copy()

    matrix = use_df.pivot_table(
        index="subject",
        columns=label_col,
        values=value_col,
        aggfunc="mean",
    )

    if matrix.empty:
        return None

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix


def save_hierarchical_clustermap(
    matrix_df,
    out_png,
    title,
    colorbar_label,
    standardize_columns=True,
    hide_subject_labels_if_gt=80,
    method="ward",
    metric="euclidean",
    figsize=None,
    xtick_fontsize=HEATMAP_XTICK_FONTSIZE,
    ytick_fontsize=HEATMAP_YTICK_FONTSIZE,
    title_fontsize=HEATMAP_TITLE_FONTSIZE,
    axis_label_fontsize=HEATMAP_AXIS_LABEL_FONTSIZE,
    cbar_fontsize=HEATMAP_CBAR_FONTSIZE,
    dpi=HEATMAP_DPI,
):
    """Save clustered heatmap plus raw/scaled matrices used to make the plot."""
    if matrix_df is None or matrix_df.empty:
        return False
    if matrix_df.shape[0] < 2 or matrix_df.shape[1] < 2:
        print(f"Skipping clustermap; need >=2 subjects and >=2 columns: {out_png}")
        return False

    raw_matrix = matrix_df.copy()
    plot_matrix = _standardize_columns(raw_matrix) if standardize_columns else raw_matrix.copy()

    if figsize is None:
        # Dynamic sizing to avoid unreadable plots.
        fig_w = min(max(12, 0.40 * plot_matrix.shape[1] + 4), 34)
        fig_h = min(max(10, 0.10 * plot_matrix.shape[0] + 4), 40)
        figsize = (fig_w, fig_h)

    show_y = plot_matrix.shape[0] <= hide_subject_labels_if_gt

    cg = sns.clustermap(
        plot_matrix,
        row_cluster=True,
        col_cluster=True,
        method=method,
        metric=metric,
        cmap="coolwarm",
        center=0,
        linewidths=0.0,
        xticklabels=True,
        yticklabels=show_y,
        figsize=figsize,
        dendrogram_ratio=(0.14, 0.14),
        cbar_kws={"label": colorbar_label},
    )

    cg.ax_heatmap.set_xlabel("SHAP / embedding / feature dimension", fontsize=axis_label_fontsize)
    cg.ax_heatmap.set_ylabel("Subjects", fontsize=axis_label_fontsize)
    cg.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=xtick_fontsize)
    if show_y:
        cg.ax_heatmap.tick_params(axis="y", labelsize=ytick_fontsize)
    else:
        cg.ax_heatmap.set_yticklabels([])

    if hasattr(cg, "cax") and cg.cax is not None:
        cg.cax.set_ylabel(colorbar_label, fontsize=cbar_fontsize)
        cg.cax.tick_params(labelsize=max(cbar_fontsize - 2, 8))

    cg.fig.suptitle(title, y=1.02, fontsize=title_fontsize)
    cg.savefig(out_png, dpi=dpi, bbox_inches="tight")

    row_order = plot_matrix.index[cg.dendrogram_row.reordered_ind]
    col_order = plot_matrix.columns[cg.dendrogram_col.reordered_ind]
    raw_clustered = raw_matrix.loc[row_order, col_order]
    scaled_clustered = plot_matrix.loc[row_order, col_order]

    base = os.path.splitext(out_png)[0]
    raw_matrix.to_csv(base + "_matrix_raw.csv")
    plot_matrix.to_csv(base + "_matrix_scaled.csv")
    raw_clustered.to_csv(base + "_clustered_matrix_raw.csv")
    scaled_clustered.to_csv(base + "_clustered_matrix_scaled.csv")

    plt.close(cg.fig)
    print(f"Saved hierarchical clustermap: {out_png}")
    return True


def categorize_global_feature(feature_name):
    """Map a global model input name to an interpretable category."""
    f = str(feature_name).lower()

    if any(tok in f for tok in ["sex", "gender", "genotype", "apoe", "race", "ethnic"]):
        return "demographics"

    if any(tok in f for tok in [
        "clustering_coeff", "clustering", "path_length", "efficiency",
        "global_efficiency", "local_efficiency", "graph", "network"
    ]):
        return "graph"

    if any(tok in f for tok in [
        "amyloid", "abeta", "tau", "ptau", "ptau217", "nfl", "gfap",
        "biomarker", "pca_", "transcript", "gene", "rna"
    ]):
        return "biomarkers"

    if any(tok in f for tok in [
        "bmi", "bp_", "blood", "systolic", "diastolic", "pulse", "map",
        "chol", "hdl", "ldl", "triglycer", "glucose", "hba1c"
    ]):
        return "cardiovascular"

    return "other"


def save_global_shap_category_heatmaps(
    cohort,
    feature_set,
    cluster_dir,
    long_df,
    plot_rows,
):
    """Save separate subject x SHAP heatmaps for global-feature categories."""
    if long_df is None or long_df.empty:
        return plot_rows
    if "feature_name" not in long_df.columns:
        return plot_rows

    long_df = long_df.copy()
    long_df["global_feature_category"] = long_df["feature_name"].apply(categorize_global_feature)
    category_csv = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_global_feature_categories.csv")
    long_df[["feature_name", "global_feature_category"]].drop_duplicates().sort_values(["global_feature_category", "feature_name"]).to_csv(category_csv, index=False)

    for category in GLOBAL_SHAP_CATEGORIES_TO_SAVE:
        sub = long_df[long_df["global_feature_category"] == category].copy()
        if sub.empty:
            continue
        matrix = _build_subject_feature_matrix(
            sub,
            label_col="feature_name",
            value_col=HEATMAP_VALUE_COLUMN,
            top_n=None,
            ranking_col="abs_SHAP",
        )
        out_png = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_global_shap_{category}_hierarchical_heatmap.png")
        ok = save_hierarchical_clustermap(
            matrix,
            out_png=out_png,
            title=f"{cohort} {feature_set}: subject × {category} SHAP heatmap",
            colorbar_label=("Scaled SHAP value" if HEATMAP_STANDARDIZE_COLUMNS else HEATMAP_VALUE_COLUMN),
            standardize_columns=HEATMAP_STANDARDIZE_COLUMNS,
            hide_subject_labels_if_gt=HEATMAP_HIDE_SUBJECT_LABELS_IF_GT,
            figsize=CATEGORY_SHAP_HEATMAP_FIGSIZE,
        )
        plot_rows.append({
            "plot": out_png,
            "type": f"global_shap_{category}_heatmap",
            "status": "saved" if ok else "skipped",
            "n_features": 0 if matrix is None else int(matrix.shape[1]),
        })

    return plot_rows


@torch.no_grad()
def _extract_embedding_from_graph(model, data, representation="gnn"):
    """
    Extract one subject-level model representation.

    representation="gnn": pooled graph embedding before global-feature fusion.
    representation="fused": pooled graph embedding + global features.
    representation="regressor_hidden": hidden layer before final output.
    """
    model.eval()
    d = data.clone().to(next(model.parameters()).device)
    if not hasattr(d, "batch") or d.batch is None:
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=d.x.device)
    if d.edge_attr.dim() == 1:
        d.edge_attr = d.edge_attr.unsqueeze(-1)

    x = F.relu(model.bn1(model.conv1(d.x.float(), d.edge_index, d.edge_attr.float())))
    x = F.relu(model.bn2(model.conv2(x, d.edge_index, d.edge_attr.float())))
    x = F.relu(model.bn3(model.conv3(x, d.edge_index, d.edge_attr.float())))
    gnn_emb = global_mean_pool(x, d.batch)

    if representation == "gnn":
        emb = gnn_emb
    else:
        if hasattr(d, "global_features") and d.global_features is not None:
            gf = d.global_features.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
        else:
            gf = torch.zeros((gnn_emb.shape[0], 0), device=gnn_emb.device)
        fused = torch.cat([gnn_emb, gf], dim=1)
        if representation == "fused":
            emb = fused
        elif representation == "regressor_hidden":
            h = fused
            # All layers except the final scalar output layer.
            for layer in list(model.regressor.children())[:-1]:
                h = layer(h)
            emb = h
        else:
            raise ValueError(f"Unknown embedding representation: {representation}")

    return emb.detach().cpu().numpy().reshape(-1)


def build_embedding_matrix(graphs, ckpt, representation="gnn"):
    """Build subject x embedding-dimension matrix from the trained model."""
    device = torch.device("cpu")
    model = build_model_from_checkpoint(ckpt, device)
    rows = []
    subject_ids = []
    for idx, g in enumerate(graphs, 1):
        subject_ids.append(safe_filename(get_graph_identifier(g, idx)))
        rows.append(_extract_embedding_from_graph(model, g, representation=representation))

    if not rows:
        return None
    mat = np.vstack(rows)
    cols = [f"embed_{i + 1:03d}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=subject_ids, columns=cols)


def make_subject_level_hierarchical_heatmaps(
    cohort,
    feature_set,
    out_root,
    global_dir,
    node_dir,
    edge_dir,
    graphs=None,
    ckpt=None,
):
    """Create subject x SHAP and subject x embedding hierarchical heatmaps."""
    cluster_dir = ensure_dir(os.path.join(out_root, "hierarchical_clustering"))
    plot_rows = []

    if RUN_GLOBAL_FEATURE_SHAP and MAKE_GLOBAL_SHAP_CLUSTER_HEATMAP:
        long_df = _load_subject_level_long_table(
            global_dir,
            file_prefix="global_feature_shap_subject_",
            label_col="feature_name",
        )
        matrix = _build_subject_feature_matrix(
            long_df,
            label_col="feature_name",
            value_col=HEATMAP_VALUE_COLUMN,
            top_n=HEATMAP_TOP_N_GLOBAL_FEATURES,
            ranking_col="abs_SHAP",
        )
        out_png = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_global_feature_shap_hierarchical_heatmap.png")
        ok = save_hierarchical_clustermap(
            matrix,
            out_png=out_png,
            title=f"{cohort} {feature_set}: subject × global SHAP heatmap",
            colorbar_label=("Scaled SHAP value" if HEATMAP_STANDARDIZE_COLUMNS else HEATMAP_VALUE_COLUMN),
            standardize_columns=HEATMAP_STANDARDIZE_COLUMNS,
            hide_subject_labels_if_gt=HEATMAP_HIDE_SUBJECT_LABELS_IF_GT,
            figsize=GLOBAL_SHAP_HEATMAP_FIGSIZE,
        )
        plot_rows.append({"plot": out_png, "type": "global_feature_shap_heatmap", "status": "saved" if ok else "skipped"})

        if MAKE_GLOBAL_CATEGORY_SHAP_HEATMAPS:
            plot_rows = save_global_shap_category_heatmaps(
                cohort=cohort,
                feature_set=feature_set,
                cluster_dir=cluster_dir,
                long_df=long_df,
                plot_rows=plot_rows,
            )

    if RUN_NODE_FEATURE_SHAP and MAKE_NODE_SHAP_CLUSTER_HEATMAP:
        long_df = _load_subject_level_long_table(
            node_dir,
            file_prefix="node_feature_shap_subject_",
            label_col="feature_name",
        )
        if long_df is not None and not long_df.empty:
            long_df = long_df.copy()
            long_df["node_feature_label"] = long_df["node_label"].astype(str) + " | " + long_df["feature_name"].astype(str)
        matrix = _build_subject_feature_matrix(
            long_df,
            label_col="node_feature_label",
            value_col=HEATMAP_VALUE_COLUMN,
            top_n=HEATMAP_TOP_N_NODE_FEATURES,
            ranking_col="abs_SHAP",
        )
        out_png = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_node_feature_shap_hierarchical_heatmap.png")
        ok = save_hierarchical_clustermap(
            matrix,
            out_png=out_png,
            title=f"{cohort} {feature_set}: subject × node-feature SHAP heatmap",
            colorbar_label=("Scaled SHAP value" if HEATMAP_STANDARDIZE_COLUMNS else HEATMAP_VALUE_COLUMN),
            standardize_columns=HEATMAP_STANDARDIZE_COLUMNS,
            hide_subject_labels_if_gt=HEATMAP_HIDE_SUBJECT_LABELS_IF_GT,
            figsize=NODE_SHAP_HEATMAP_FIGSIZE,
        )
        plot_rows.append({"plot": out_png, "type": "node_feature_shap_heatmap", "status": "saved" if ok else "skipped"})

    if RUN_EDGE_SHAP and MAKE_EDGE_SHAP_CLUSTER_HEATMAP:
        long_df = _load_subject_level_long_table(
            edge_dir,
            file_prefix="edge_shap_subject_",
            label_col="Edge",
        )
        if long_df is not None and not long_df.empty:
            long_df = long_df.copy()
            if "edge_feature_name" in long_df.columns and long_df["edge_feature_name"].nunique() > 1:
                long_df["edge_label_full"] = long_df["Edge"].astype(str) + " | " + long_df["edge_feature_name"].astype(str)
            else:
                long_df["edge_label_full"] = long_df["Edge"].astype(str)
        matrix = _build_subject_feature_matrix(
            long_df,
            label_col="edge_label_full",
            value_col=HEATMAP_VALUE_COLUMN,
            top_n=HEATMAP_TOP_N_EDGE_FEATURES,
            ranking_col="abs_SHAP",
        )
        out_png = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_edge_shap_hierarchical_heatmap.png")
        ok = save_hierarchical_clustermap(
            matrix,
            out_png=out_png,
            title=f"{cohort} {feature_set}: subject × edge SHAP heatmap",
            colorbar_label=("Scaled SHAP value" if HEATMAP_STANDARDIZE_COLUMNS else HEATMAP_VALUE_COLUMN),
            standardize_columns=HEATMAP_STANDARDIZE_COLUMNS,
            hide_subject_labels_if_gt=HEATMAP_HIDE_SUBJECT_LABELS_IF_GT,
            figsize=EDGE_SHAP_HEATMAP_FIGSIZE,
        )
        plot_rows.append({"plot": out_png, "type": "edge_shap_heatmap", "status": "saved" if ok else "skipped"})

    if MAKE_EMBEDDING_CLUSTER_HEATMAP and graphs is not None and ckpt is not None:
        try:
            emb_matrix = build_embedding_matrix(graphs, ckpt, representation=EMBEDDING_REPRESENTATION)
            out_png = os.path.join(cluster_dir, f"{cohort.lower()}_{feature_set}_{EMBEDDING_REPRESENTATION}_embedding_hierarchical_heatmap.png")
            ok = save_hierarchical_clustermap(
                emb_matrix,
                out_png=out_png,
                title=f"{cohort} {feature_set}: subject × {EMBEDDING_REPRESENTATION} embedding heatmap",
                colorbar_label=("Scaled embedding value" if EMBEDDING_STANDARDIZE_COLUMNS else "Embedding value"),
                standardize_columns=EMBEDDING_STANDARDIZE_COLUMNS,
                hide_subject_labels_if_gt=HEATMAP_HIDE_SUBJECT_LABELS_IF_GT,
                figsize=EMBEDDING_HEATMAP_FIGSIZE,
            )
            plot_rows.append({"plot": out_png, "type": f"{EMBEDDING_REPRESENTATION}_embedding_heatmap", "status": "saved" if ok else "skipped"})
        except Exception as e:
            err_path = os.path.join(cluster_dir, "embedding_heatmap_error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            plot_rows.append({"plot": err_path, "type": "embedding_heatmap", "status": "failed", "error": str(e)})

    if plot_rows:
        pd.DataFrame(plot_rows).to_csv(os.path.join(cluster_dir, "hierarchical_heatmap_log.csv"), index=False)
    return plot_rows


# =========================================================
# MAIN RUNNERS
# =========================================================
def run_one(cohort, feature_set):
    paths = get_input_paths(cohort, feature_set)
    out_root = ensure_dir(paths["out_root"])
    edge_dir = ensure_dir(os.path.join(out_root, "edge_shap"))
    node_dir = ensure_dir(os.path.join(out_root, "node_feature_shap"))
    global_dir = ensure_dir(os.path.join(out_root, "global_feature_shap"))
    region_dir = ensure_dir(os.path.join(out_root, "region_from_edge_shap"))

    print("\n" + "=" * 100)
    print(f"SHAP | cohort={cohort} | feature_set={feature_set}")
    print("=" * 100)
    print("Graph:", paths["graph_path"])
    print("Model:", paths["model_path"])
    print("Encoding:", paths["encoding_info_path"])
    print("Output:", out_root)

    run_summary = {
        "cohort": cohort,
        "feature_set": feature_set,
        "status": "started",
        "graph_path": paths["graph_path"],
        "model_path": paths["model_path"],
        "encoding_info_path": paths["encoding_info_path"],
        "out_root": out_root,
        "n_graphs": np.nan,
        "global_status_ok_or_exists": np.nan,
        "node_status_ok_or_exists": np.nan,
        "edge_status_ok_or_exists": np.nan,
        "error": None,
    }

    try:
        graphs, ckpt, device, node_feature_names, global_feature_names, edge_feature_names, node_label_map = load_graphs_and_model(paths)
        run_summary["n_graphs"] = len(graphs)
        run_summary["node_feat_dim"] = int(ckpt["node_feat_dim"])
        run_summary["global_feat_dim"] = int(ckpt["global_feat_dim"])
        run_summary["edge_dim"] = int(ckpt["edge_dim"])

        print("Loaded graphs:", len(graphs))
        print("Dims:", {"node": ckpt["node_feat_dim"], "global": ckpt["global_feat_dim"], "edge": ckpt["edge_dim"]})
        print("Global feature names:", global_feature_names)

        save_feature_name_summary(out_root, node_feature_names, global_feature_names, edge_feature_names)

        subject_summaries = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(compute_one_subject)(
                idx,
                data,
                ckpt,
                node_feature_names,
                global_feature_names,
                edge_feature_names,
                node_label_map,
                edge_dir,
                node_dir,
                global_dir,
            )
            for idx, data in enumerate(graphs, 1)
        )

        summary_df = pd.DataFrame(subject_summaries)
        summary_csv = os.path.join(out_root, "input_shap_summary_all_subjects.csv")
        summary_df.to_csv(summary_csv, index=False)
        print("Saved subject-level SHAP run summary:", summary_csv)

        edge_summary = summarize_edge_shap(edge_dir) if RUN_EDGE_SHAP else None
        global_summary = summarize_global_feature_shap(global_dir) if RUN_GLOBAL_FEATURE_SHAP else None
        node_summary = summarize_node_feature_shap(node_dir) if RUN_NODE_FEATURE_SHAP else None
        region_summary = summarize_region_from_edge_shap(edge_summary, region_dir) if RUN_EDGE_SHAP else None

        make_plots(cohort, feature_set, out_root, edge_summary, global_summary, node_summary, region_summary)
        heatmap_rows = make_subject_level_hierarchical_heatmaps(
            cohort=cohort,
            feature_set=feature_set,
            out_root=out_root,
            global_dir=global_dir,
            node_dir=node_dir,
            edge_dir=edge_dir,
            graphs=graphs,
            ckpt=ckpt,
        )
        run_summary["n_hierarchical_heatmaps_saved"] = int(sum(r.get("status") == "saved" for r in heatmap_rows))

        run_summary["status"] = "ok"
        if "global_status" in summary_df.columns:
            run_summary["global_status_ok_or_exists"] = int(summary_df["global_status"].isin(["ok", "exists"]).sum())
        if "node_status" in summary_df.columns:
            run_summary["node_status_ok_or_exists"] = int(summary_df["node_status"].isin(["ok", "exists"]).sum())
        if "edge_status" in summary_df.columns:
            run_summary["edge_status_ok_or_exists"] = int(summary_df["edge_status"].isin(["ok", "exists"]).sum())

        # Save a compact top-feature table for quick comparison.
        compact_rows = []
        if global_summary is not None:
            tmp = global_summary.head(TOP_N_GLOBAL_FEATURES).copy()
            tmp["category"] = "global"
            tmp = tmp.rename(columns={"feature_name": "feature"})
            compact_rows.append(tmp[["category", "feature", "mean_abs_SHAP", "mean_SHAP", "n_subjects"]])
        if node_summary is not None:
            tmp = node_summary.head(TOP_N_NODE_FEATURES).copy()
            tmp["category"] = "node"
            tmp["feature"] = tmp["node_label"].astype(str) + " | " + tmp["feature_name"].astype(str)
            compact_rows.append(tmp[["category", "feature", "mean_abs_SHAP", "mean_SHAP", "n_subjects"]])
        if edge_summary is not None:
            tmp = edge_summary.head(TOP_N_EDGES).copy()
            tmp["category"] = "edge"
            tmp["feature"] = tmp["Edge"].astype(str)
            compact_rows.append(tmp[["category", "feature", "mean_abs_SHAP", "mean_SHAP", "n_subjects"]])
        if compact_rows:
            compact = pd.concat(compact_rows, ignore_index=True)
            compact.insert(0, "cohort", cohort)
            compact.insert(1, "feature_set", feature_set)
            compact.to_csv(os.path.join(out_root, "top_shap_features_compact.csv"), index=False)

    except Exception as e:
        run_summary["status"] = "failed"
        run_summary["error"] = str(e)
        with open(os.path.join(out_root, "shap_error_traceback.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("FAILED:", cohort, feature_set)
        print(str(e))

    pd.DataFrame([run_summary]).to_csv(os.path.join(out_root, "shap_run_summary.csv"), index=False)
    return run_summary


def main():
    all_rows = []
    for cohort in COHORTS_TO_RUN:
        if cohort not in COHORT_CONFIG:
            print(f"Skipping unsupported cohort: {cohort}")
            continue
        for feature_set in FEATURE_SETS_TO_RUN:
            row = run_one(cohort, feature_set)
            all_rows.append(row)
            combined = pd.DataFrame(all_rows)
            combined.to_csv(os.path.join(OUT_BASE, "combined_shap_run_summary.csv"), index=False)

    combined = pd.DataFrame(all_rows)
    combined_csv = os.path.join(OUT_BASE, "combined_shap_run_summary.csv")
    combined_xlsx = os.path.join(OUT_BASE, "combined_shap_run_summary.xlsx")
    combined.to_csv(combined_csv, index=False)
    try:
        combined.to_excel(combined_xlsx, index=False)
    except Exception:
        pass

    print("\n" + "=" * 100)
    print("SHAP ALL COHORTS / ABLATIONS FINISHED")
    print("Combined summary:", combined_csv)
    print("Output base:", OUT_BASE)
    print("=" * 100)


if __name__ == "__main__":
    main()
