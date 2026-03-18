#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:20:40 2026

@author: ines
"""
import os
import pandas as pd
WORK = os.environ["WORK"]


# ========= RUTAS =========
clusters_path = os.path.join(
    WORK, "ines/results/shap_hc_sensitivity_analysis/k_4/merged_embeddings_metadata_with_hc_clusters_k4.csv"
)

data6_path = os.path.join(
    WORK, "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics.xlsx"
)


# ========= CARGAR =========
df_clusters = pd.read_csv(clusters_path)
df_data6    = pd.read_excel(data6_path)

# ========= QUEDARTE SOLO CON LAS COLUMNAS NECESARIAS =========
# Sacamos únicamente las claves de unión + la columna de cluster
df_clusters_small = df_clusters[["SUBJ_key", "MRI_Exam_key", "Cluster_HC"]].copy()

# Por si hay duplicados en el CSV de clusters
df_clusters_small = df_clusters_small.drop_duplicates(subset=["SUBJ_key", "MRI_Exam_key"])

# ========= UNIR =========
df_merged = df_data6.merge(
    df_clusters_small,
    on=["SUBJ_key", "MRI_Exam_key"],
    how="left"
)

# ========= COMPROBAR =========
print("Filas en data6 original:", len(df_data6))
print("Filas en data6 unido:", len(df_merged))
print("Clusters no encontrados:", df_merged["Cluster_HC"].isna().sum())

# ========= GUARDAR =========
out_xlsx= os.path.join(
    WORK, "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics_plusCluster.xlsx"
)
out_csv= os.path.join(
    WORK, "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics_plusCluster.csv"
)


df_merged.to_excel(out_xlsx, index=False)
df_merged.to_csv(out_csv, index=False)

print("Guardado en:")
print(out_xlsx)
print(out_csv)