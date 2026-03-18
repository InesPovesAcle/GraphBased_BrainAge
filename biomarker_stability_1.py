import os
import pandas as pd

WORK = os.environ["WORK"]
RESULTS_DIR = os.path.join(WORK, "ines/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# LOAD EXCEL PRINCIPAL
# =========================
excel_path = os.path.join(WORK, "ines/data/AD_DECODE_data6.xlsx")
df = pd.read_excel(excel_path, sheet_name=0)

if "MRI_Exam" not in df.columns:
    raise ValueError("No encuentro la columna 'MRI_Exam' en AD_DECODE_data6.xlsx")

# =========================
# LOAD CSV PREDICCIONES
# =========================
pred_path = os.path.join(WORK, "ines/results/brain_age_predictions_with_metadata.csv")
df_preds = pd.read_csv(pred_path)

if "MRI_Exam_fixed" not in df_preds.columns:
    raise ValueError("No encuentro 'MRI_Exam_fixed' en brain_age_predictions_with_metadata.csv")
if "cBAG" not in df_preds.columns:
    raise ValueError("No encuentro 'cBAG' en brain_age_predictions_with_metadata.csv")

# =========================
# NORMALIZAR IDs (00123 -> 123)
# =========================
df_preds["MRI_Exam_int"] = pd.to_numeric(df_preds["MRI_Exam_fixed"], errors="coerce").astype("Int64")
df["MRI_Exam_int"] = pd.to_numeric(df["MRI_Exam"], errors="coerce").astype("Int64")

# 1 fila por sujeto
df_cbag = (
    df_preds.dropna(subset=["MRI_Exam_int"])
           .groupby("MRI_Exam_int", as_index=False)["cBAG"]
           .mean()
)

# Merge
df = df.merge(df_cbag, on="MRI_Exam_int", how="left")

print("Merge done. Missing cBAG:", df["cBAG"].isna().sum())
print("N con cBAG:", df["cBAG"].notna().sum())

# Guardar
out_path = os.path.join(RESULTS_DIR, "AD_DECODE_data6_with_cBAG.xlsx")
df.to_excel(out_path, index=False)
print("Saved:", out_path)

