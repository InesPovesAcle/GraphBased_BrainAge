import os
import re
import pandas as pd
import numpy as np

print("ADNI CONNECTOMES (conn_plain only)\n")

base_dir = os.path.join(os.environ["WORK"], "ines/data/harmonization/ADNI/connectomes/DWI/plain")

conn_plain_paths = []
for root, _, files in os.walk(base_dir):
    for fn in files:
        if fn.endswith("_conn_plain.csv"):
            conn_plain_paths.append(os.path.join(root, fn))

print(f"Found conn_plain files: {len(conn_plain_paths)}")

# Parse keys -> RID + y
keys = []
rids = []
ys = []

bad = 0
for p in conn_plain_paths:
    fn = os.path.basename(p).replace("_conn_plain.csv", "")
    m = re.match(r"^R(\d+)_y(\d+)$", fn)
    if not m:
        bad += 1
        continue
    rid = int(m.group(1))
    y = int(m.group(2))
    keys.append(f"R{rid}_y{y}")
    rids.append(rid)
    ys.append(y)

connectome_index = pd.DataFrame({"CONN_KEY": keys, "RID": rids, "Y": ys}).drop_duplicates()

print(f"Valid conn_plain entries: {connectome_index.shape[0]}")
if bad:
    print(f"Skipped (unexpected filename pattern): {bad}")

# --- COUNT SUBJECTS like AD-DECODE (unique subjects) ---
n_subjects = connectome_index["RID"].nunique()
print(f"\nUnique subjects (RID): {n_subjects}")

# --- counts per timepoint ---
tp_counts = connectome_index["Y"].value_counts().sort_index()
print("\nConnectomes per timepoint (Y):")
for y, n in tp_counts.items():
    print(f"y{y}: {n}")

# --- subjects with baseline / y4 etc ---
subs_y0 = set(connectome_index.loc[connectome_index["Y"] == 0, "RID"])
subs_y4 = set(connectome_index.loc[connectome_index["Y"] == 4, "RID"])

print(f"\nSubjects with y0: {len(subs_y0)}")
print(f"Subjects with y4: {len(subs_y4)}")
print(f"Subjects with BOTH y0 & y4: {len(subs_y0 & subs_y4)}")

# Optional: show how many timepoints per subject
tp_per_subj = connectome_index.groupby("RID")["Y"].nunique().value_counts().sort_index()
print("\n#timepoints per subject:")
for k, v in tp_per_subj.items():
    print(f"{k} timepoints: {v} subjects")


# ==========================
# METADATA MATCH (by RID only)
# ==========================
print("\nADNI METADATA (match by RID only)\n")
import numpy as np
metadata_path = os.path.join(
    os.environ["WORK"],
    "ines/data/harmonization/ADNI/metadata/FINAL_idaSearch_3_19_2025.xlsx"
)

df_metadata = pd.read_excel(metadata_path, sheet_name="METADATA")
print(f"Metadata loaded: {df_metadata.shape[0]} rows")

# Extract RID from Subject ID: 002_S_0413 -> 413
df_metadata["RID"] = (
    df_metadata["Subject ID"].astype(str).str.extract(r"_(\d+)$").astype(float).astype("Int64")
)

df_metadata_clean = df_metadata.dropna(subset=["RID"]).copy()

# Unique metadata subjects
n_meta_subs = df_metadata_clean["RID"].nunique()
print(f"Unique subjects in metadata (RID): {n_meta_subs}")

# Match subjects
conn_rids = set(connectome_index["RID"])
meta_rids = set(df_metadata_clean["RID"].astype(int))

matched_rids = conn_rids & meta_rids
print(f"\nMatched subjects (metadata & connectome): {len(matched_rids)} out of {len(conn_rids)}")

# If you want a metadata dataframe with only matched subjects:
df_matched_subjects = df_metadata_clean[df_metadata_clean["RID"].astype(int).isin(matched_rids)].copy()

# and make it ONE row per subject (like AD-DECODE)
df_matched_subjects_uniq = df_matched_subjects.drop_duplicates(subset=["RID"]).copy()
print(f"Matched subjects (unique rows): {df_matched_subjects_uniq.shape[0]}")




# Usa 1 fila por sujeto (como AD-DECODE)
df = df_matched_subjects_uniq.copy()

# -------------------------
# Columnas por letra (0-index)
# A=0, B=1, C=2, D=3, E=4, ... J=9
# -------------------------
sex_col = df.columns[2]   # C
dx_col  = df.columns[4]   # E
age_col = df.columns[9]   # J

print("Usando columnas:")
print("  SEX:", sex_col)
print("  DX :", dx_col)
print("  AGE:", age_col)

# -------------------------
# Edad
# -------------------------
df["age_num"] = pd.to_numeric(df[age_col], errors="coerce")

age_mean = df["age_num"].mean()
age_std  = df["age_num"].std()
age_min  = df["age_num"].min()
age_max  = df["age_num"].max()

# -------------------------
# Sexo (robusto)
# -------------------------
sex_raw = df[sex_col].astype(str).str.strip()

def normalize_sex(x):
    s = str(x).strip().lower()
    if s in ["m", "male", "hombre", "1"]:
        return "Male (M)"
    if s in ["f", "female", "mujer", "2"]:
        return "Female (F)"
    # si ya viene "Male"/"Female" o similar, intenta inferir
    if s.startswith("m"):
        return "Male (M)"
    if s.startswith("f"):
        return "Female (F)"
    return "Unknown"

df["Sex_Label"] = sex_raw.apply(normalize_sex)

# -------------------------
# Diagnóstico (como categoría)
# -------------------------
df["DX_Label"] = df[dx_col].astype(str).str.strip()
df.loc[df["DX_Label"].isin(["nan", "None", ""]), "DX_Label"] = np.nan

# -------------------------
# APOE genotype desde apoea1 + apoea2
# -------------------------
# Si no existen exactamente esos nombres, busca algo parecido:
apoe1_col = "apoea1" if "apoea1" in df.columns else next((c for c in df.columns if "apoe" in c.lower() and "1" in c), None)
apoe2_col = "apoea2" if "apoea2" in df.columns else next((c for c in df.columns if "apoe" in c.lower() and "2" in c), None)

print("APOE columns:", apoe1_col, apoe2_col)

if apoe1_col is not None and apoe2_col is not None:
    a1 = pd.to_numeric(df[apoe1_col], errors="coerce")
    a2 = pd.to_numeric(df[apoe2_col], errors="coerce")

    # ordenar alelos para que 4/3 y 3/4 sean iguales -> 34
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)

    df["APOE_genotype"] = np.where(
        lo.notna() & hi.notna(),
        "APOE" + lo.astype(int).astype(str) + hi.astype(int).astype(str),
        np.nan
    )
else:
    df["APOE_genotype"] = np.nan
    print("WARNING: No encuentro columnas apoea1/apoea2 (o equivalentes).")

# -------------------------
# Conteos y porcentajes
# -------------------------
def counts_perc(series):
    counts = series.value_counts(dropna=True)
    perc = (counts / len(series) * 100).round(1)
    return counts, perc

dx_counts, dx_perc = counts_perc(df["DX_Label"])
sex_counts, sex_perc = counts_perc(df["Sex_Label"])
apoe_counts, apoe_perc = counts_perc(df["APOE_genotype"])

# -------------------------
# Print estilo AD-DECODE
# -------------------------
print("\n=== AGE ===")
print(f"Mean ± SD : {age_mean:.2f} ± {age_std:.2f}")
print(f"Range     : [{age_min:.1f}, {age_max:.1f}]")

print("\n=== DIAGNOSTIC GROUP ===")
for grp, n in dx_counts.items():
    print(f"{grp:<15}: {n:3d} ({dx_perc[grp]}%)")

print("\n=== SEX ===")
for sx, n in sex_counts.items():
    print(f"{sx:<10}: {n:3d} ({sex_perc[sx]}%)")

print("\n=== APOE GENOTYPE ===")
for gt, n in apoe_counts.items():
    print(f"{gt:<7}: {n:3d} ({apoe_perc[gt]}%)")

# Tabla resumen (opcional)
rows = [
    ["Age", "Mean ± SD", f"{age_mean:.2f} ± {age_std:.2f}"],
    ["Age", "Range",     f"[{age_min:.1f}, {age_max:.1f}]"],
]
rows += [["Diagnostic group", g, f"{dx_counts[g]} ({dx_perc[g]}%)"] for g in dx_counts.index]
rows += [["Sex", s, f"{sex_counts[s]} ({sex_perc[s]}%)"] for s in sex_counts.index]
rows += [["APOE genotype", a, f"{apoe_counts[a]} ({apoe_perc[a]}%)"] for a in apoe_counts.index]

df_summary = pd.DataFrame(rows, columns=["Category", "Value", "Count (%)"])
print("\n--- SUMMARY TABLE (ADNI; matched subjects) ---")
print(df_summary)
