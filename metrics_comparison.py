import pandas as pd
import re
import os

RESULTS_DIR = os.path.join(os.environ["WORK"], "ines/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rows = []

# =========================================================
# 1) ADNI: raw + bias-corrected
# =========================================================
path_adni_raw_corrected = "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADNI/final_metrics_raw_vs_corrected.txt"

with open(path_adni_raw_corrected, "r", encoding="utf-8") as f:
    text = f.read()

# RAW
raw_mae_match = re.search(r"FINAL METRICS \(RAW\).*?MAE\s+([0-9.]+)", text, re.S)
raw_rmse_match = re.search(r"FINAL METRICS \(RAW\).*?RMSE\s+([0-9.]+)", text, re.S)
raw_r2_match = re.search(r"FINAL METRICS \(RAW\).*?R2\s+([0-9.]+)", text, re.S)

# BIAS-CORRECTED
corr_mae_match = re.search(r"FINAL METRICS \(BIAS-CORRECTED\).*?MAE\s+([0-9.]+)", text, re.S)
corr_rmse_match = re.search(r"FINAL METRICS \(BIAS-CORRECTED\).*?RMSE\s+([0-9.]+)", text, re.S)
corr_r2_match = re.search(r"FINAL METRICS \(BIAS-CORRECTED\).*?R2\s+([0-9.]+)", text, re.S)

rows.append({
    "model": "ADNI",
    "dataset": "ADNI",
    "subjects": "all",
    "bias_correction": False,
    "pca": None,
    "mae": float(raw_mae_match.group(1)),
    "rmse": float(raw_rmse_match.group(1)),
    "r2": float(raw_r2_match.group(1))
})

rows.append({
    "model": "ADNI",
    "dataset": "ADNI",
    "subjects": "all",
    "bias_correction": True,
    "pca": None,
    "mae": float(corr_mae_match.group(1)),
    "rmse": float(corr_rmse_match.group(1)),
    "r2": float(corr_r2_match.group(1))
})

# =========================================================
# 2) AD-DECODE all subjects (bias corrected)
# =========================================================
path_addecode_all_bias = "/mnt/newStor/paros/paros_WORK/ines/results/results_with_bias_correction_all_subjects/model_performance_metrics.txt"

with open(path_addecode_all_bias, "r", encoding="utf-8") as f:
    text = f.read()

mae_match = re.search(r"MAE:\s*([0-9.]+)", text)
rmse_match = re.search(r"RMSE:\s*([0-9.]+)", text)
r2_match = re.search(r"R2:\s*([0-9.]+)", text)

rows.append({
    "model": "AD-DECODE",
    "dataset": "AD-DECODE",
    "subjects": "all",
    "bias_correction": True,
    "pca": None,
    "mae": float(mae_match.group(1)),
    "rmse": float(rmse_match.group(1)),
    "r2": float(r2_match.group(1))
})

# =========================================================
# 3) AD-DECODE healthy (CV mean)
# =========================================================
path_addecode_healthy_cv = "/mnt/newStor/paros/paros_WORK/ines/results/results_with_bias_correction/metrics_per_fold_repeat.csv"

df_cv = pd.read_csv(path_addecode_healthy_cv)

rows.append({
    "model": "AD-DECODE",
    "dataset": "AD-DECODE",
    "subjects": "healthy",
    "bias_correction": False,
    "pca": True,
    "mae": df_cv["MAE"].mean(),
    "rmse": df_cv["RMSE"].mean(),
    "r2": df_cv["R2"].mean()
})

# =========================================================
# 4) AD-DECODE healthy without PCA
# =========================================================
path_addecode_no_pca = "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionHealthy_withoutPCA/metrics_per_fold_repeat.csv"

df_no_pca = pd.read_csv(path_addecode_no_pca)
df_no_pca.columns = [c.strip() for c in df_no_pca.columns]

rows.append({
    "model": "AD-DECODE_noPCA",
    "dataset": "AD-DECODE",
    "subjects": "healthy",
    "bias_correction": False,
    "pca": False,
    "mae": df_no_pca["MAE"].mean(),
    "rmse": df_no_pca["RMSE"].mean(),
    "r2": df_no_pca["R2"].mean()
})
# =========================================================
# 5) AD-DECODE healthy pretrained ADNI model (finetuned)
# =========================================================
path_addecode_pretrained = "/mnt/newStor/paros/paros_WORK/ines/results/addecode_training_with_adni_model/metrics_by_fold_repeat_addecode_pretrained_with_ADNImodel_FINETUNED1.csv"

df_pretrained = pd.read_csv(path_addecode_pretrained)

df_pretrained.columns = [c.strip() for c in df_pretrained.columns]

rows.append({
    "model": "AD-DECODE_pretrained_ADNI_finetuned",
    "dataset": "AD-DECODE",
    "subjects": "healthy",
    "bias_correction": False,
    "pca": True,
    "mae": df_pretrained["MAE"].mean(),
    "rmse": df_pretrained["RMSE"].mean(),
    "r2": df_pretrained["R2"].mean()
})


#a mano porque si no no me da tiempo
rows.append({
    "model": "AD-DECODE_pretrained_ADNI_finetuned",
    "dataset": "AD-DECODE",
    "subjects": "healthy",
    "bias_correction": True,
    "pca": False,
    "mae": 7.02,
    "rmse": 8.67,
    "r2": 0.65
})

# =========================================================
# 6) ADDECODE model with contrastive learning embeddings
# =========================================================

path_addecode_cl = "/mnt/newStor/paros/paros_WORK/ines/results/Model_with_shap_embeddings/ADDECODE_CLmodel_metrics.csv"

df_cl = pd.read_csv(path_addecode_cl)

rows.append({
    "model": "ADDECODE_model_with_contrastive_learning_embeddings",
    "dataset": "AD-DECODE",
    "subjects": "healthy",
    "bias_correction": False,
    "pca": True,
    "mae": df_cl["MAE_mean"].iloc[0],
    "rmse": df_cl["RMSE_mean"].iloc[0],
    "r2": df_cl["R2_mean"].iloc[0]
})
# =========================================================
# FINAL TABLE
# =========================================================
df_final = pd.DataFrame(rows)

df_final = df_final[
    [
        "model", 
        "subjects",
        "bias_correction",
        "pca",
        "mae",
        "rmse",
        "r2",
    ]
]

print(df_final)


df_final["mae"] = df_final["mae"].round(4)
df_final["rmse"] = df_final["rmse"].round(4)
df_final["r2"] = df_final["r2"].round(4)


output_path = os.path.join(RESULTS_DIR, "model_metrics_comparison.csv")

df_final.to_csv(output_path, index=False)

print("Saved metrics table to:", output_path)