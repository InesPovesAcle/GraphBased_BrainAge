#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:22:32 2026

@author: ines
"""

import os
import pandas as pd

WORK = os.environ["WORK"]

embed_path = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings.csv"
)

meta_path = os.path.join(
    WORK,
    "ines/data/AD_DECODE_data4.xlsx"
)

out_path = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings_with_riskprob.csv"
)

df_embed = pd.read_csv(embed_path)
df_meta = pd.read_excel(meta_path)

df_embed["Subject_ID"] = df_embed["Subject_ID"].astype(str).str.zfill(5)
df_meta["MRI_Exam_fixed"] = (
    df_meta["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# Elige aquí las columnas que realmente existan en tu metadata
cols_to_add = ["MRI_Exam_fixed"]
for col in ["risk_for_ad", "Risk", "APOE", "genotype", "sex", "age"]:
    if col in df_meta.columns:
        cols_to_add.append(col)

df_out = df_embed.merge(
    df_meta[cols_to_add],
    left_on="Subject_ID",
    right_on="MRI_Exam_fixed",
    how="left"
)

df_out.to_csv(out_path, index=False)
print("Saved:", out_path)