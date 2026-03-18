import os
import pandas as pd

# =========================
# LOAD
# =========================

WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
RESULTS_DIR = os.path.join(WORK, "ines/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

metadata_path = os.path.join(WORK, "ines/data/AD_DECODE_data6.xlsx")
metadata = pd.read_excel(metadata_path)

withPCA_path = os.path.join(WORK, "ines/results/brain_age_predictions_with_metadata.csv")
cBAGwithPCA = pd.read_csv(withPCA_path)

withoutPCA_path = os.path.join(
    WORK,
    "ines/results/BrainAgePredictionAll_withoutPCA/brain_age_predictions_all_without_PCA.csv"
)
CBAGwithoutPCA = pd.read_csv(withoutPCA_path)

pca_path = os.path.join(WORK, "ines/data/PCA_human_blood_top30.csv")
pca = pd.read_csv(pca_path)

out_path = os.path.join(RESULTS_DIR, "AD_DECODE_data6_with_cBAGs_and_PCA.xlsx")


# =========================
# HELPERS
# =========================
def norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

def dedup_key(df, key, name):
    # Prevent many-to-many merges that duplicate metadata rows
    ndup = df.duplicated(key, keep=False).sum()
    if ndup > 0:
        print(f"[WARNING] {name}: {ndup} rows have duplicated key '{key}'. Keeping first occurrence per key.")
    return df.drop_duplicates(subset=[key], keep="first")


# =========================
# CHECK REQUIRED COLUMNS
# =========================
require_cols(metadata, ["MRI_Exam", "IDRNA"], "metadata")
require_cols(cBAGwithPCA, ["MRI_Exam_fixed", "cBAG"], "cBAGwithPCA")
require_cols(CBAGwithoutPCA, ["MRI_Exam_fixed", "cBAG"], "CBAGwithoutPCA")

# pca join columns (adjust here if your PCA file uses different names)
# expected: pca has "ID"
require_cols(pca, ["ID"], "pca")


# =========================
# NORMALIZE JOIN KEYS
# =========================
metadata["MRI_Exam_norm"] = norm_id(metadata["MRI_Exam"])
metadata["IDRNA_norm"] = norm_id(metadata["IDRNA"])

cBAGwithPCA["MRI_Exam_fixed_norm"] = norm_id(cBAGwithPCA["MRI_Exam_fixed"])
CBAGwithoutPCA["MRI_Exam_fixed_norm"] = norm_id(CBAGwithoutPCA["MRI_Exam_fixed"])

pca["ID_norm"] = norm_id(pca["ID"])


# =========================
# PREPARE cBAG TABLES (1 row per exam)
# =========================
with_tbl = cBAGwithPCA[["MRI_Exam_fixed_norm", "cBAG"]].rename(columns={"cBAG": "cBAG_withPCA"})
with_tbl = dedup_key(with_tbl, "MRI_Exam_fixed_norm", "cBAGwithPCA")

without_tbl = CBAGwithoutPCA[["MRI_Exam_fixed_norm", "cBAG"]].rename(columns={"cBAG": "cBAG_withoutPCA"})
without_tbl = dedup_key(without_tbl, "MRI_Exam_fixed_norm", "CBAGwithoutPCA")


# =========================
# MERGE 1: metadata + cBAGs (by MRI exam)
# =========================
merged = metadata.merge(
    with_tbl,
    left_on="MRI_Exam_norm",
    right_on="MRI_Exam_fixed_norm",
    how="left",
    validate="m:1"
).merge(
    without_tbl,
    left_on="MRI_Exam_norm",
    right_on="MRI_Exam_fixed_norm",
    how="left",
    validate="m:1"
)

# drop helper key columns from intermediate merges
merged = merged.drop(columns=["MRI_Exam_fixed_norm_x", "MRI_Exam_fixed_norm_y"], errors="ignore")


# =========================
# PREPARE PCA TABLE (1 row per ID)
# =========================
# keep all PCA columns; if you only want a subset, filter here
pca_tbl = dedup_key(pca, "ID_norm", "pca")

# =========================
# MERGE 2: add PCA (by RNA ID)
# =========================
final = merged.merge(
    pca_tbl,
    left_on="IDRNA_norm",
    right_on="ID_norm",
    how="left",
    validate="m:1"
)

# =========================
# FINAL CLEANUP + SANITY CHECKS
# =========================
# Remove normalization helper cols
final = final.drop(columns=["MRI_Exam_norm", "IDRNA_norm", "ID_norm"], errors="ignore")

# Ensure we did not duplicate metadata rows
if len(final) != len(metadata):
    raise RuntimeError(
        f"Row count changed after merges! metadata={len(metadata)} final={len(final)}. "
        f"This usually means duplicate keys in one of the join tables."
    )

# Report match coverage
print("--- Match summary ---")
print("metadata rows:", len(metadata))
print("cBAG_withPCA matched:", final["cBAG_withPCA"].notna().sum())
print("cBAG_withoutPCA matched:", final["cBAG_withoutPCA"].notna().sum())

# PCA match coverage: at least one PCA numeric col matched (or any col besides ID)
pca_cols = [c for c in pca_tbl.columns if c not in ["ID", "ID_norm"]]
if pca_cols:
    print("PCA matched (any PCA field present):", final[pca_cols].notna().any(axis=1).sum())
else:
    print("PCA matched: (no extra PCA columns found besides ID)")

# =========================
# SAVE
# =========================
final.to_excel(out_path, index=False)
print("Saved:", out_path)

# normalize IDs first (important)
metadata_ids = metadata["IDRNA"].astype("string").str.strip()
pca_ids = pca["ID"].astype("string").str.strip()

# total subjects with MRI exams
n_mri_subjects = metadata_ids.notna().sum()

# subjects with MRI exams AND PCA
mask_match = metadata_ids.isin(pca_ids)
n_matches = mask_match.sum()

# subjects with MRI exams but NO PCA
n_no_match = (~mask_match).sum()

print("Total subjects with MRI_Exam:", n_mri_subjects)
print("Subjects with PCA match:", n_matches)
print("Subjects without PCA:", n_no_match)
print("Match percentage:", f"{n_matches/n_mri_subjects:.2%}")


# =========================
# NORMALIZE IDS
# =========================
metadata["IDRNA_norm"] = metadata["IDRNA"].astype("string").str.strip()
pca["ID_norm"] = pca["ID"].astype("string").str.strip()

# =========================
# MATCHING SUBJECTS
# =========================
matching_mask = metadata["IDRNA_norm"].isin(pca["ID_norm"])
matching_subjects = metadata.loc[matching_mask].copy()

print("Number of matching subjects:", len(matching_subjects))
print("\nMatching subjects preview:")
print(matching_subjects[["IDRNA"]].head(20))


# =========================
# METADATA SUBJECTS WITHOUT PCA
# =========================
excluded_metadata = metadata.loc[~matching_mask].copy()

print("\nNumber of metadata subjects WITHOUT PCA:", len(excluded_metadata))
print("\nExcluded metadata preview:")
print(excluded_metadata[["IDRNA"]].head(20))


# =========================
# PCA SUBJECTS NOT IN METADATA
# =========================
pca_only_mask = ~pca["ID_norm"].isin(metadata["IDRNA_norm"])
excluded_pca = pca.loc[pca_only_mask].copy()

print("\nNumber of PCA subjects NOT in metadata:", len(excluded_pca))
print("\nExcluded PCA preview:")
print(excluded_pca[["ID"]].head(20))


# =========================
# OPTIONAL: SAVE THESE FILES
# =========================
matching_subjects.to_excel(
    os.path.join(RESULTS_DIR, "matching_subjects_metadata_PCA.xlsx"),
    index=False
)

excluded_metadata.to_excel(
    os.path.join(RESULTS_DIR, "metadata_subjects_without_PCA.xlsx"),
    index=False
)

excluded_pca.to_excel(
    os.path.join(RESULTS_DIR, "PCA_subjects_not_in_metadata.xlsx"),
    index=False
)

print("\nSaved matching and excluded subject files.")