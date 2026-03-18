import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress 
import numpy as np
# =========================
# CONFIG / PATHS
# =========================
WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
RESULTS_DIR = os.path.join(WORK, "ines/results/merged_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

metadata_path = os.path.join(WORK, "ines/data/AD_DECODE_data6.xlsx")
withPCA_path = os.path.join(WORK, "ines/results/brain_age_predictions_with_metadata.csv")
withoutPCA_path = os.path.join(
    WORK, "ines/results/BrainAgePredictionAll_withoutPCA/brain_age_predictions_all_without_PCA.csv"
)
pca_path = os.path.join(WORK, "ines/data/PCA_human_blood_top30.csv")

out_path = os.path.join(RESULTS_DIR, "AD_DECODE_data6_with_cBAGs_and_PCA.xlsx")

# =========================
# LOAD
# =========================
metadata = pd.read_excel(metadata_path)
cBAGwithPCA = pd.read_csv(withPCA_path)
CBAGwithoutPCA = pd.read_csv(withoutPCA_path)
pca = pd.read_csv(pca_path)

# =========================
# HELPERS
# =========================
def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

def dedup_key(df, key, name):
    ndup = df.duplicated(key, keep=False).sum()
    if ndup > 0:
        print(f"[WARNING] {name}: {ndup} duplicated rows for key '{key}'. Keeping first per key.")
    return df.drop_duplicates(subset=[key], keep="first")

def norm_exam_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()

def canonical_subject_id(x) -> str:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper()
    s = re.sub(r"[\s_\-\.]", "", s)  # remove separators
    m = re.search(r"ADDECODE", s)
    if m:
        s = s[m.start():]
    return s

def summarize_key_overlap(meta_keys: pd.Series, pca_keys: pd.Series, label=""):
    meta_set = set(meta_keys.dropna())
    pca_set = set(pca_keys.dropna())
    inter = meta_set & pca_set
    print(f"\n--- Key overlap {label} ---")
    print("Unique metadata keys:", len(meta_set))
    print("Unique PCA keys:", len(pca_set))
    print("Unique intersection:", len(inter))
    if len(meta_set) > 0:
        print("Intersection % of metadata:", f"{len(inter)/len(meta_set):.2%}")
    if len(pca_set) > 0:
        print("Intersection % of PCA:", f"{len(inter)/len(pca_set):.2%}")

# =========================
# CHECK REQUIRED COLUMNS
# =========================
require_cols(metadata, ["MRI_Exam", "IDRNA"], "metadata")
require_cols(cBAGwithPCA, ["MRI_Exam_fixed", "cBAG"], "cBAGwithPCA")
require_cols(CBAGwithoutPCA, ["MRI_Exam_fixed", "cBAG"], "CBAGwithoutPCA")
require_cols(pca, ["ID"], "pca")  # <-- PCA ID column

# =========================
# NORMALIZE JOIN KEYS
# =========================
metadata["MRI_Exam_key"] = norm_exam_id(metadata["MRI_Exam"])
cBAGwithPCA["MRI_Exam_fixed_key"] = norm_exam_id(cBAGwithPCA["MRI_Exam_fixed"])
CBAGwithoutPCA["MRI_Exam_fixed_key"] = norm_exam_id(CBAGwithoutPCA["MRI_Exam_fixed"])

metadata["SUBJ_key"] = metadata["IDRNA"].map(canonical_subject_id)
pca["SUBJ_key"] = pca["ID"].map(canonical_subject_id)

# Check that PCA.ID corresponds to metadata.IDRNA (via canonical key overlap)
summarize_key_overlap(metadata["SUBJ_key"], pca["SUBJ_key"], label="metadata.IDRNA vs pca.ID (raw)")

# =========================
# PREPARE cBAG TABLES
# =========================
with_tbl = cBAGwithPCA[["MRI_Exam_fixed_key", "cBAG"]].rename(columns={"cBAG": "cBAG_withPCA"})
with_tbl = dedup_key(with_tbl, "MRI_Exam_fixed_key", "cBAGwithPCA")

without_tbl = CBAGwithoutPCA[["MRI_Exam_fixed_key", "cBAG"]].rename(columns={"cBAG": "cBAG_withoutPCA"})
without_tbl = dedup_key(without_tbl, "MRI_Exam_fixed_key", "CBAGwithoutPCA")

# =========================
# MERGE 1: metadata + cBAGs
# =========================
merged = metadata.merge(
    with_tbl,
    left_on="MRI_Exam_key",
    right_on="MRI_Exam_fixed_key",
    how="left",
    validate="m:1"
).merge(
    without_tbl,
    left_on="MRI_Exam_key",
    right_on="MRI_Exam_fixed_key",
    how="left",
    validate="m:1"
)

merged = merged.drop(columns=["MRI_Exam_fixed_key_x", "MRI_Exam_fixed_key_y"], errors="ignore")

# =========================
# PREPARE PCA TABLE + MERGE 2
# =========================
pca_tbl = dedup_key(pca, "SUBJ_key", "pca")
pca_tbl["_PCA_PRESENT"] = True

summarize_key_overlap(metadata["SUBJ_key"], pca_tbl["SUBJ_key"], label="metadata.IDRNA vs pca.ID (after dedup)")

final = merged.merge(
    pca_tbl,
    on="SUBJ_key",
    how="left",
    validate="m:1",
    suffixes=("", "_PCA")
)

# =========================
# SANITY CHECKS
# =========================
if len(final) != len(metadata):
    raise RuntimeError(
        f"Row count changed after merges! metadata={len(metadata)} final={len(final)}. "
        f"This indicates duplicate keys in a join table."
    )

# Build a SAFE boolean mask Series (fixes your KeyError)
pca_match_mask = final["_PCA_PRESENT"].fillna(False).astype(bool)

# =========================
# REPORTS
# =========================
print("\n--- Match summary ---")
print("metadata rows:", len(metadata))
print("cBAG_withPCA matched:", int(final["cBAG_withPCA"].notna().sum()))
print("cBAG_withoutPCA matched:", int(final["cBAG_withoutPCA"].notna().sum()))
print("PCA matched (ID -> IDRNA, canonical):", int(pca_match_mask.sum()))
print("PCA match percentage:", f"{pca_match_mask.mean():.2%}")

# Examples of mismatches
pca_keys_set = set(pca_tbl["SUBJ_key"].dropna())
missing_pca = final.loc[~final["SUBJ_key"].isin(pca_keys_set), ["IDRNA", "SUBJ_key"]].head(20)
print("\nExamples metadata subjects WITHOUT PCA match (first 20):")
print(missing_pca)

meta_keys_set = set(metadata["SUBJ_key"].dropna())
pca_only = pca_tbl.loc[~pca_tbl["SUBJ_key"].isin(meta_keys_set), ["ID", "SUBJ_key"]].head(20)
print("\nExamples PCA subjects NOT in metadata (first 20):")
print(pca_only)

# =========================
# SAVE MAIN OUTPUT
# =========================
final.to_excel(out_path, index=False)
print("\nSaved:", out_path)

# =========================
# SAVE MATCHED / EXCLUDED FILES (fixed)
# =========================
matching_subjects = final.loc[pca_match_mask].copy()
excluded_metadata = final.loc[~pca_match_mask].copy()
excluded_pca = pca_tbl.loc[~pca_tbl["SUBJ_key"].isin(metadata["SUBJ_key"])].copy()

matching_subjects.to_excel(os.path.join(RESULTS_DIR, "matching_subjects_metadata_PCA.xlsx"), index=False)
excluded_metadata.to_excel(os.path.join(RESULTS_DIR, "metadata_subjects_without_PCA.xlsx"), index=False)
excluded_pca.to_excel(os.path.join(RESULTS_DIR, "PCA_subjects_not_in_metadata.xlsx"), index=False)

print("Saved matching/excluded subject files.")

plot_path = os.path.join(
    RESULTS_DIR,
    "scatter_cBAG_with_vs_without_PCA.png"
)
excel_path = os.path.join(
    RESULTS_DIR,
    "AD_DECODE_data6_with_cBAGs_and_PCA.xlsx"
    )
# =========================
# LOAD DATA
# =========================
df = pd.read_excel(excel_path)

# Keep only subjects with both cBAGs
mask = (
    df["cBAG_withPCA"].notna() &
    df["cBAG_withoutPCA"].notna()
)

data = df.loc[mask].copy()

x = data["cBAG_withoutPCA"].values
y = data["cBAG_withPCA"].values

# =========================
# REGRESSION
# =========================
reg = linregress(x, y)

beta = reg.slope
intercept = reg.intercept
r = reg.rvalue
r2 = r**2
p = reg.pvalue
n = len(x)

print("N:", n)
print("beta:", beta)
print("R²:", r2)
print("p-value:", p)

# =========================
# PLOT
# =========================
plt.figure(figsize=(6,6))

# scatter
plt.scatter(x, y)

# regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = intercept + beta * x_line
plt.plot(x_line, y_line)

# identity line (optional but useful)
plt.plot(x_line, x_line, linestyle="--")

plt.xlabel("cBAG without PCA")
plt.ylabel("cBAG with PCA")

plt.title(
    f"cBAG with PCA vs without PCA\n"
    f"N={n}, R²={r2:.3f}, beta={beta:.3f}, p={p:.3e}"
)

plt.tight_layout()

# =========================
# SAVE FIGURE
# =========================
plt.savefig(plot_path, dpi=300)

plt.show()

print("Plot saved to:", plot_path)