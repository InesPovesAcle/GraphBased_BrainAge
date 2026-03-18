#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:06:00 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build edge-level SHAP matrix for Option A

Input:
    Multiple files like:
    edge_shap_subject_<SUBJECT_ID>.csv

Each subject file must contain at least:
    Node_i, Node_j, SHAP_val

Output:
    One CSV with:
    - 1 row per subject
    - 1 Subject_ID column
    - 3486 edge SHAP columns (for 84 nodes)

Expected output shape:
    (n_subjects, 3487)
"""

import os
import glob
import pandas as pd
import numpy as np

WORK = os.environ["WORK"]

# Carpeta donde están tus edge SHAP por sujeto
INPUT_DIR = os.path.join(
    WORK,
    "ines/results/Shap_edges/edges_addecode"
)

# Archivo final que necesita Option A
OUTPUT_CSV = os.path.join(
    INPUT_DIR,
    "edge_shap_matrix_all_subjects.csv"
)

print("Input dir:", INPUT_DIR)
print("Output csv:", OUTPUT_CSV)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def canonical_edge_name(i, j):
    i, j = sorted((int(i), int(j)))
    return f"edge_{i}_{j}"

# Lista esperada de 3486 edges para 84 nodos
expected_edges = [canonical_edge_name(i, j) for i in range(84) for j in range(i + 1, 84)]

print("Expected number of edges:", len(expected_edges))  # 3486

# -------------------------------------------------------------------
# Find subject files
# -------------------------------------------------------------------

pattern = os.path.join(INPUT_DIR, "edge_shap_subject_*.csv")
subject_files = sorted(glob.glob(pattern))

if len(subject_files) == 0:
    raise FileNotFoundError(f"No files found matching: {pattern}")

print("Subject files found:", len(subject_files))

# -------------------------------------------------------------------
# Build one row per subject
# -------------------------------------------------------------------

rows = []

for fpath in subject_files:
    fname = os.path.basename(fpath)
    subject_id = fname.replace("edge_shap_subject_", "").replace(".csv", "")

    df = pd.read_csv(fpath)

    required_cols = {"Node_i", "Node_j", "SHAP_val"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{fname} is missing required columns: {missing}")

    # Build edge names
    df["edge_name"] = [
        canonical_edge_name(i, j)
        for i, j in zip(df["Node_i"], df["Node_j"])
    ]

    # Keep only first occurrence if duplicates exist
    df = df.drop_duplicates(subset="edge_name", keep="first")

    # Create dict with all expected edges initialized to NaN
    row = {"Subject_ID": str(subject_id).zfill(5)}
    row.update({edge: np.nan for edge in expected_edges})

    # Fill SHAP values
    for _, r in df.iterrows():
        edge = r["edge_name"]
        if edge in row:
            row[edge] = r["SHAP_val"]

    rows.append(row)

# -------------------------------------------------------------------
# Final dataframe
# -------------------------------------------------------------------

df_out = pd.DataFrame(rows)

# Ensure exact column order
df_out = df_out[["Subject_ID"] + expected_edges]

print("Output shape:", df_out.shape)

# Sanity check: should be n_subjects x 3487
if df_out.shape[1] != 3487:
    raise RuntimeError(
        f"Unexpected number of columns: {df_out.shape[1]} "
        f"(expected 3487 = Subject_ID + 3486 edges)"
    )

# Report missing values
missing_total = df_out.isna().sum().sum()
print("Total missing values:", missing_total)

# Save
df_out.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
print("DONE.")