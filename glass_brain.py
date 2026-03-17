#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:59:00 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from nilearn import plotting

# =========================================================
# PATHS
# =========================================================

WORK = os.environ["WORK"]

# Folder with your already-generated SHAP edge CSVs
SHAP_DIR = os.path.join(WORK, "ines/results/Shap_edges/edges_addecode")

# Output folder for glass-brain figures
OUT_DIR = os.path.join(WORK, "ines/results/Shap_edges/glass_brain_addecode")
os.makedirs(OUT_DIR, exist_ok=True)

# Atlas + lookup paths
# CHANGE THESE IF YOUR FILES ARE ELSEWHERE
ATLAS_NII = os.path.join(WORK, "ines/data/atlas/IITmean_RPI_labels.nii.gz")
LOOKUP_XLSX = os.path.join(WORK, "ines/data/atlas/IITmean_RPI_index.xlsx")

print(f"Reading SHAP CSVs from: {SHAP_DIR}")
print(f"Saving figures to: {OUT_DIR}")
print(f"Atlas path: {ATLAS_NII}")
print(f"Lookup path: {LOOKUP_XLSX}")

# =========================================================
# REGION NAMES
# =========================================================

region_names = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

if len(region_names) != 84:
    raise ValueError(f"Expected 84 region names, got {len(region_names)}")

# =========================================================
# 1) LOAD ALL SHAP CSVs
# =========================================================

all_shap_dfs = []

for fname in os.listdir(SHAP_DIR):
    if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(SHAP_DIR, fname))
        df.columns = df.columns.str.strip()
        all_shap_dfs.append(df)

if len(all_shap_dfs) == 0:
    raise RuntimeError(f"No SHAP CSVs found in {SHAP_DIR}")

shap_df = pd.concat(all_shap_dfs, ignore_index=True)
print(f"Loaded SHAP rows: {len(shap_df)}")

# =========================================================
# 2) REPLACE NODE INDICES WITH REGION NAMES
# =========================================================

shap_df["Region_1"] = shap_df["Node_i"].apply(lambda x: region_names[int(x)])
shap_df["Region_2"] = shap_df["Node_j"].apply(lambda x: region_names[int(x)])


shap_df["Region_1"] = shap_df["Region_1"].astype(str).str.strip().str.replace('"', "", regex=False)
shap_df["Region_2"] = shap_df["Region_2"].astype(str).str.strip().str.replace('"', "", regex=False)

# =========================================================
# 3) GROUP AND AVERAGE SHAP PER EDGE
# =========================================================

top_n = 10
use_abs = True  # keeps your labels later if you want

grouped = (
    shap_df.groupby(["Region_1", "Region_2"])["SHAP_val"]
    .agg(
        mean_SHAP="mean",
        mean_abs_SHAP=lambda x: np.mean(np.abs(x))
    )
    .reset_index()
)

# choose top edges by absolute importance
grouped = grouped.sort_values(by="mean_abs_SHAP", ascending=False).reset_index(drop=True)

# BUT paint them by signed SHAP
grouped["PlotWeight"] = grouped["mean_SHAP"]

top_df = grouped.head(top_n).copy()
top_df["Connection"] = top_df["Region_1"] + " ↔ " + top_df["Region_2"]

print("\nTop connections:")
print(top_df[["Connection", "mean_SHAP", "mean_abs_SHAP"]])
# =========================================================
# 4) LOAD REGION CENTROIDS FROM ATLAS
# =========================================================

if not os.path.exists(ATLAS_NII):
    raise FileNotFoundError(f"Atlas not found: {ATLAS_NII}")

if not os.path.exists(LOOKUP_XLSX):
    raise FileNotFoundError(f"Lookup file not found: {LOOKUP_XLSX}")

img = nib.load(ATLAS_NII)
data = img.get_fdata()
affine = img.affine

region_labels = np.unique(data)
region_labels = region_labels[region_labels != 0]

centroids = []
for label in region_labels:
    coords = np.argwhere(data == label)
    center_voxel = coords.mean(axis=0)
    center_mni = nib.affines.apply_affine(affine, center_voxel)
    centroids.append(center_mni)

centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
centroid_df["Label"] = region_labels.astype(int)

lookup = pd.read_excel(LOOKUP_XLSX)
lookup.columns = lookup.columns.str.strip()

# Clean structure names
lookup["Structure"] = (
    lookup["Structure"]
    .astype(str)
    .str.strip()
    .str.replace('"', "", regex=False)
)

required_cols = {"index2", "Structure"}
if not required_cols.issubset(set(lookup.columns)):
    raise ValueError(
        f"Lookup file must contain columns {required_cols}. "
        f"Found: {list(lookup.columns)}"
    )


centroid_df["Label"] = pd.to_numeric(centroid_df["Label"], errors="coerce").astype(int)

lookup["index2"] = pd.to_numeric(lookup["index2"], errors="coerce")

final_df = pd.merge(
    centroid_df,
    lookup,
    left_on="Label",
    right_on="index2",
    how="inner"
)

region_name_to_coords = {
    str(row["Structure"]).strip().replace('"', ""): [row["X"], row["Y"], row["Z"]]
    for _, row in final_df.iterrows()
}

print("Number of atlas regions matched:", len(region_name_to_coords))
print("Example atlas region names:", list(region_name_to_coords.keys())[:10])

# =========================================================
# 5) BUILD CONNECTIVITY MATRIX FOR TOP EDGES
# =========================================================

regions_involved = list(set(top_df["Region_1"]) | set(top_df["Region_2"]))

missing_regions = [r for r in regions_involved if r not in region_name_to_coords]
if len(missing_regions) > 0:
    raise KeyError(
        "These regions are missing from the atlas lookup:\n" + "\n".join(missing_regions)
    )

region_to_index = {region: idx for idx, region in enumerate(regions_involved)}
coords = [region_name_to_coords[region] for region in regions_involved]

n = len(regions_involved)
con_matrix = np.zeros((n, n))

for _, row in top_df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    con_matrix[i, j] = row["PlotWeight"]
    con_matrix[j, i] = row["PlotWeight"]

# =========================================================
# 6) PLOT GLASS BRAIN
# =========================================================
node_colors = "silver"


edge_max = np.max(np.abs(con_matrix))

display = plotting.plot_connectome(
    con_matrix,
    coords,
    edge_threshold="0%",
    node_color="silver",
    node_size=50,
    edge_cmap=plt.cm.bwr,
    edge_vmin=-edge_max,
    edge_vmax=edge_max,
    colorbar=True,
    title="Top SHAP connections"
)

out_png = os.path.join(OUT_DIR, "glass_brain_dti_top10_addecode_signed.png")
display.savefig(out_png)
plt.show()

print(f"\nSaved figure: {out_png}")



# =========================================================
# 7) TOP CONNECTIONS BY HEMISPHERE (what your PI asked for)
# =========================================================

print("\nBuilding top connections by hemisphere...")

TABLES_OUT_DIR = os.path.join(OUT_DIR, "top30_by_hemisphere")
os.makedirs(TABLES_OUT_DIR, exist_ok=True)

def get_hemi(region_name):
    r = str(region_name).strip().replace('"', "")
    if r.startswith("ctx-lh-") or r.startswith("Left-"):
        return "Left"
    if r.startswith("ctx-rh-") or r.startswith("Right-"):
        return "Right"
    return "Unknown"

def classify_connection(region1, region2):
    h1 = get_hemi(region1)
    h2 = get_hemi(region2)

    if h1 == "Left" and h2 == "Left":
        return "Left"
    elif h1 == "Right" and h2 == "Right":
        return "Right"
    elif {h1, h2} == {"Left", "Right"}:
        return "Interhemispheric"
    else:
        return "Unknown"

# Build edge-level table from grouped results
hemisphere_df = grouped.copy()

hemisphere_df["Region_1"] = hemisphere_df["Region_1"].astype(str).str.strip().str.replace('"', "", regex=False)
hemisphere_df["Region_2"] = hemisphere_df["Region_2"].astype(str).str.strip().str.replace('"', "", regex=False)
hemisphere_df["Connection"] = hemisphere_df["Region_1"] + " ↔ " + hemisphere_df["Region_2"]

hemisphere_df["Connection_Type"] = hemisphere_df.apply(
    lambda row: classify_connection(row["Region_1"], row["Region_2"]),
    axis=1
)

hemisphere_df = hemisphere_df[hemisphere_df["Connection_Type"] != "Unknown"].copy()

print("\nCounts by type:")
print(hemisphere_df["Connection_Type"].value_counts())

# Select top 10 per category
top_left = (
    hemisphere_df[hemisphere_df["Connection_Type"] == "Left"]
    .sort_values("mean_abs_SHAP", ascending=False)
    .head(10)
    .copy()
)

top_right = (
    hemisphere_df[hemisphere_df["Connection_Type"] == "Right"]
    .sort_values("mean_abs_SHAP", ascending=False)
    .head(10)
    .copy()
)

top_inter = (
    hemisphere_df[hemisphere_df["Connection_Type"] == "Interhemispheric"]
    .sort_values("mean_abs_SHAP", ascending=False)
    .head(10)
    .copy()
)

top30 = pd.concat([top_left, top_right, top_inter], ignore_index=True)

# Reorder columns
cols_to_keep = ["Connection_Type", "Region_1", "Region_2", "Connection", "PlotWeight"]
extra_cols = [c for c in ["mean_abs_SHAP", "mean_SHAP"] if c in top30.columns]
top30 = top30[["Connection_Type", "Region_1", "Region_2", "Connection"] + extra_cols + ["PlotWeight"]]

# Save tables
top30_csv = os.path.join(TABLES_OUT_DIR, "top30_edges_by_hemisphere.csv")
top_left_csv = os.path.join(TABLES_OUT_DIR, "top10_left_edges.csv")
top_right_csv = os.path.join(TABLES_OUT_DIR, "top10_right_edges.csv")
top_inter_csv = os.path.join(TABLES_OUT_DIR, "top10_interhemispheric_edges.csv")

top30.to_csv(top30_csv, index=False)
top_left.to_csv(top_left_csv, index=False)
top_right.to_csv(top_right_csv, index=False)
top_inter.to_csv(top_inter_csv, index=False)

print(f"Saved table: {top30_csv}")
print(f"Saved table: {top_left_csv}")
print(f"Saved table: {top_right_csv}")
print(f"Saved table: {top_inter_csv}")

# =========================================================
# GLASS BRAINS BY HEMISPHERE
# =========================================================

print("\nPlotting hemisphere-specific glass brains...")

brain_sets = [
    (top_left, "Left hemisphere connections"),
    (top_right, "Right hemisphere connections"),
    (top_inter, "Interhemispheric connections")
]

for df_set, title in brain_sets:

    if df_set.empty:
        print(f"No connections for {title}")
        continue

    regions = list(set(df_set["Region_1"]) | set(df_set["Region_2"]))
    region_to_index = {r: i for i, r in enumerate(regions)}

    coords_subset = [region_name_to_coords[r] for r in regions]

    n = len(regions)
    con_matrix_subset = np.zeros((n, n))

    for _, row in df_set.iterrows():
        i = region_to_index[row["Region_1"]]
        j = region_to_index[row["Region_2"]]

        con_matrix_subset[i, j] = row["PlotWeight"]
        con_matrix_subset[j, i] = row["PlotWeight"]

    edge_max = np.max(np.abs(con_matrix_subset))

    display = plotting.plot_connectome(
        con_matrix_subset,
        coords_subset,
        edge_threshold="0%",
        node_color="silver",
        node_size=50,
        edge_cmap=plt.cm.bwr,
        edge_vmin=-edge_max,
        edge_vmax=edge_max,
        colorbar=True,
        title=title
    )

    # nombres claros para guardar
    if "Left" in title:
        fname = "glass_brain_top_left.png"
    elif "Right" in title:
        fname = "glass_brain_top_right.png"
    else:
        fname = "glass_brain_top_interhemispheric.png"

    out_file = os.path.join(TABLES_OUT_DIR, fname)

    # guardar exactamente el glass brain que se ve
    display.savefig(out_file)
    print(f"Saved glass brain: {out_file}")

    plt.show()
# ---------------------------------------------------------
# Plot combined figure (max 30)
# ---------------------------------------------------------

plot_df = top30.copy()

plot_df["Connection_Type"] = pd.Categorical(
    plot_df["Connection_Type"],
    categories=["Left", "Right", "Interhemispheric"],
    ordered=True
)

plot_df = plot_df.sort_values(
    ["Connection_Type", "PlotWeight"],
    ascending=[True, True]
)

color_map = {
    "Left": "steelblue",
    "Right": "darkorange",
    "Interhemispheric": "seagreen"
}
colors = plot_df["Connection_Type"].map(color_map)

# ---------------------------------------------------------
# Plot combined figure (max 30)
# ---------------------------------------------------------

plt.figure(figsize=(12, 12))
plt.barh(plot_df["Connection"], plot_df["PlotWeight"], color=colors)
plt.xlabel("Mean |SHAP|" if use_abs else "|Mean SHAP|")
plt.ylabel("Connection")
plt.title("Top SHAP Connections by Hemisphere (Left / Right / Interhemispheric)")
plt.tight_layout()

top30_png = os.path.join(TABLES_OUT_DIR, "top30_edges_by_hemisphere.png")
plt.savefig(top30_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {top30_png}")

# ---------------------------------------------------------
# Plot 3-panel figure: Left / Right / Interhemispheric
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharex=False)

panel_info = [
    (top_left.sort_values("PlotWeight", ascending=True), "Left hemisphere", "steelblue", axes[0]),
    (top_right.sort_values("PlotWeight", ascending=True), "Right hemisphere", "darkorange", axes[1]),
    (top_inter.sort_values("PlotWeight", ascending=True), "Interhemispheric", "seagreen", axes[2]),
]

for df_panel, title, color, ax in panel_info:
    if df_panel.empty:
        ax.text(0.5, 0.5, "No connections", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        continue

    ax.barh(df_panel["Connection"], df_panel["PlotWeight"], color=color)
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP|" if use_abs else "|Mean SHAP|")
    ax.tick_params(axis="y", labelsize=9)

fig.suptitle("Top SHAP Connections by Hemisphere", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])

panel_png = os.path.join(TABLES_OUT_DIR, "top30_edges_by_hemisphere_3panels.png")
plt.savefig(panel_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {panel_png}")

# =========================================================
# 8) PAPER-READY TABLE
# =========================================================

print("\nBuilding paper-ready table...")

paper_df = top30.copy()

# Rename columns nicely
paper_df = paper_df.rename(columns={
    "Connection_Type": "Hemisphere_Group",
    "Region_1": "Region_1",
    "Region_2": "Region_2",
    "Connection": "Connection",
    "PlotWeight": "Mean_Abs_SHAP"
})

# Keep only clean columns
keep_cols = ["Hemisphere_Group", "Region_1", "Region_2", "Connection", "Mean_Abs_SHAP"]
paper_df = paper_df[keep_cols].copy()

# Order groups
paper_df["Hemisphere_Group"] = pd.Categorical(
    paper_df["Hemisphere_Group"],
    categories=["Left", "Right", "Interhemispheric"],
    ordered=True
)

paper_df = paper_df.sort_values(
    ["Hemisphere_Group", "Mean_Abs_SHAP"],
    ascending=[True, False]
).reset_index(drop=True)

# Round values for readability
paper_df["Mean_Abs_SHAP"] = paper_df["Mean_Abs_SHAP"].round(6)

# Add rank within each hemisphere group
paper_df["Rank_within_group"] = (
    paper_df.groupby("Hemisphere_Group").cumcount() + 1
)

# Reorder columns
paper_df = paper_df[
    ["Hemisphere_Group", "Rank_within_group", "Region_1", "Region_2", "Connection", "Mean_Abs_SHAP"]
]

# Save
paper_csv = os.path.join(TABLES_OUT_DIR, "paper_table_top30_edges_by_hemisphere.csv")
paper_xlsx = os.path.join(TABLES_OUT_DIR, "paper_table_top30_edges_by_hemisphere.xlsx")

paper_df.to_csv(paper_csv, index=False)
paper_df.to_excel(paper_xlsx, index=False)

print(f"Saved paper CSV: {paper_csv}")
print(f"Saved paper Excel: {paper_xlsx}")

# Print preview
print("\nPaper-ready table preview:")
print(paper_df.head(15))
