#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:43:38 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu


# =========================================================
# BASE PATH
# =========================================================
BASE = Path("/mnt/newStor/paros/paros_WORK/ines/data")

# =========================================================
# DATASET CONFIG
# =========================================================
DATASETS = {
    "ADNI": {
        "out_dir": BASE / "Regional_stats/ADNI",
        "abs_file": BASE / "Regional_stats/ADNI/ADNI_studywide_stats_BrainAbs.csv",
        "pct_file": BASE / "Regional_stats/ADNI/ADNI_studywide_stats_BrainPct.csv",
    },
    "ADDecode": {
        "out_dir": BASE / "Regional_stats/ADDecode",
        "abs_file": BASE / "Regional_stats/ADDecode/ADDecode_studywide_stats_BrainAbs.csv",
        "pct_file": BASE / "Regional_stats/ADDecode/ADDecode_studywide_stats_BrainPct.csv",
    },
    "ADRC": {
        "out_dir": BASE / "Regional_stats/ADRC",
        "abs_file": BASE / "Regional_stats/ADRC/ADRC_studywide_stats_BrainAbs.csv",
        "pct_file": BASE / "Regional_stats/ADRC/ADRC_studywide_stats_BrainPct.csv",
    },
    "HABS": {
        "out_dir": BASE / "Regional_stats/HABS",
        "abs_file": BASE / "Regional_stats/HABS/HABS_studywide_stats_BrainAbs.csv",
        "pct_file": BASE / "Regional_stats/HABS/HABS_studywide_stats_BrainPct.csv",
    },
}



RESULTS_DIR = BASE / "Regional_stats" / "_cohort_volume_stats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# REGIONS TO ANALYZE
# =========================================================
# Brain is a direct row.
# Hippocampus / Caudate / Cerebellum are combined left + right.
REGION_SPECS = {
    "Brain": {
        "rows": ["Brain"],
        "combine": "single",
    },
    "Hippocampus": {
        "rows": ["Left_Hippocampus", "Right_Hippocampus"],
        "combine": "sum",
    },
    "Caudate": {
        "rows": ["Left_Caudate", "Right_Caudate"],
        "combine": "sum",
    },
    "Cerebellum": {
        "rows": ["Left_Cerebellum_Cortex", "Right_Cerebellum_Cortex"],
        "combine": "sum",
    },
}


# =========================================================
# HELPERS
# =========================================================
def read_table_auto(path: str | Path) -> pd.DataFrame:
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except Exception:
            pass

    raise ValueError(f"Could not read file: {path}")


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def find_structure_col(df: pd.DataFrame) -> str:
    for c in ["structure", "Structure", "label", "Label", "name", "Name"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find structure column. Columns: {list(df.columns)}")


def get_subject_cols(df: pd.DataFrame, structure_col: str) -> list[str]:
    meta_cols = {df.columns[0], structure_col, "Index2", "index", "ROI", "roi"}
    out = []
    for c in df.columns:
        if c in meta_cols:
            continue
        s = str(c)
        if any(ch.isdigit() for ch in s):
            out.append(c)
    return out


def normalize_structure_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def get_row_values(df: pd.DataFrame, structure_col: str, structure_name: str, subject_cols: list[str]) -> pd.Series:
    target = normalize_structure_name(structure_name)
    normalized_col = df[structure_col].astype(str).map(normalize_structure_name)

    rows = df.loc[normalized_col == target]
    if rows.empty:
        available = df[structure_col].astype(str).dropna().unique().tolist()
        similar = [x for x in available if "hipp" in x.lower() or "amyg" in x.lower() or "caud" in x.lower() or "cereb" in x.lower()]
        raise ValueError(
            f"Row '{structure_name}' not found.\n"
            f"Examples of similar rows in this dataset: {similar[:20]}"
        )

    vals = pd.to_numeric(rows.iloc[0][subject_cols], errors="coerce")
    vals.index = subject_cols
    return vals


def extract_region_values(df: pd.DataFrame, region_spec: dict) -> pd.Series:
    structure_col = find_structure_col(df)
    subject_cols = get_subject_cols(df, structure_col)

    if region_spec["combine"] == "single":
        return get_row_values(df, structure_col, region_spec["rows"][0], subject_cols)

    if region_spec["combine"] == "sum":
        series_list = [get_row_values(df, structure_col, row_name, subject_cols) for row_name in region_spec["rows"]]
        out = series_list[0].copy()
        for s in series_list[1:]:
            out = out.add(s, fill_value=np.nan)
        return out

    raise ValueError(f"Unknown combine mode: {region_spec['combine']}")


def describe_series(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
            "iqr": np.nan, "min": np.nan, "max": np.nan,
            "q1": np.nan, "q3": np.nan
        }

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "median": float(x.median()),
        "iqr": float(q3 - q1),
        "min": float(x.min()),
        "max": float(x.max()),
        "q1": float(q1),
        "q3": float(q3),
    }


def cohens_d_welch(x1: pd.Series, x2: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce").dropna().values
    x2 = pd.to_numeric(x2, errors="coerce").dropna().values
    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    n1 = len(x1)
    n2 = len(x2)

    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(x1) - np.mean(x2)) / pooled)


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pd.Series(pvals, dtype=float)
    n = p.notna().sum()
    out = pd.Series(np.nan, index=p.index, dtype=float)

    if n == 0:
        return out

    valid = p.dropna().sort_values()
    ranks = np.arange(1, len(valid) + 1)
    adj = valid * len(valid) / ranks
    adj = np.minimum.accumulate(adj.iloc[::-1])[::-1]
    adj = adj.clip(upper=1.0)
    out.loc[adj.index] = adj.values
    return out


def make_long_df(dataset_tables: dict[str, pd.DataFrame], metric_label: str) -> pd.DataFrame:
    rows = []
    for cohort, df in dataset_tables.items():
        for region_name, region_spec in REGION_SPECS.items():
            vals = extract_region_values(df, region_spec)
            for subj, value in vals.items():
                rows.append({
                    "cohort": cohort,
                    "subject_col": subj,
                    "region": region_name,
                    "metric_type": metric_label,
                    "value": pd.to_numeric(value, errors="coerce"),
                })
    long_df = pd.DataFrame(rows)
    return long_df


def save_boxplot(long_df_region: pd.DataFrame, out_png: Path, title: str, ylabel: str):
    cohorts = list(long_df_region["cohort"].dropna().unique())
    data = [
        pd.to_numeric(
            long_df_region.loc[long_df_region["cohort"] == cohort, "value"],
            errors="coerce"
        ).dropna().values
        for cohort in cohorts
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=cohorts, showfliers=False)
    for i, cohort in enumerate(cohorts, start=1):
        y = pd.to_numeric(
            long_df_region.loc[long_df_region["cohort"] == cohort, "value"],
            errors="coerce"
        ).dropna().values
        if len(y) > 0:
            x = np.random.normal(loc=i, scale=0.04, size=len(y))
            ax.plot(x, y, "o", alpha=0.35, markersize=3)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_histogram_grid(long_df_region: pd.DataFrame, out_png: Path, title: str, xlabel: str):
    cohorts = list(long_df_region["cohort"].dropna().unique())
    n = len(cohorts)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, cohort in zip(axes, cohorts):
        vals = pd.to_numeric(
            long_df_region.loc[long_df_region["cohort"] == cohort, "value"],
            errors="coerce"
        ).dropna().values

        ax.hist(vals, bins=20)
        ax.set_title(cohort)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[len(cohorts):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_within_cohort_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric_type, region, cohort), subdf in long_df.groupby(["metric_type", "region", "cohort"]):
        stats = describe_series(subdf["value"])
        rows.append({
            "metric_type": metric_type,
            "region": region,
            "cohort": cohort,
            **stats
        })
    return pd.DataFrame(rows).sort_values(["metric_type", "region", "cohort"])


def compute_between_cohort_stats(long_df: pd.DataFrame):
    omnibus_rows = []
    pairwise_rows = []

    for (metric_type, region), subdf in long_df.groupby(["metric_type", "region"]):
        cohort_vals = {}
        for cohort, cdf in subdf.groupby("cohort"):
            vals = pd.to_numeric(cdf["value"], errors="coerce").dropna()
            if len(vals) > 0:
                cohort_vals[cohort] = vals

        usable = {k: v for k, v in cohort_vals.items() if len(v) >= 2}

        if len(usable) >= 2:
            try:
                f_stat, f_p = f_oneway(*usable.values())
            except Exception:
                f_stat, f_p = np.nan, np.nan

            try:
                kw_stat, kw_p = kruskal(*usable.values())
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
        else:
            f_stat, f_p, kw_stat, kw_p = np.nan, np.nan, np.nan, np.nan

        omnibus_rows.append({
            "metric_type": metric_type,
            "region": region,
            "n_cohorts": len(usable),
            "anova_F": f_stat,
            "anova_p": f_p,
            "kruskal_H": kw_stat,
            "kruskal_p": kw_p,
        })

        for c1, c2 in combinations(sorted(cohort_vals.keys()), 2):
            x1 = cohort_vals[c1]
            x2 = cohort_vals[c2]

            if len(x1) >= 2 and len(x2) >= 2:
                try:
                    t_stat, t_p = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
                except Exception:
                    t_stat, t_p = np.nan, np.nan

                try:
                    u_stat, u_p = mannwhitneyu(x1, x2, alternative="two-sided")
                except Exception:
                    u_stat, u_p = np.nan, np.nan

                d = cohens_d_welch(x1, x2)
            else:
                t_stat, t_p, u_stat, u_p, d = np.nan, np.nan, np.nan, np.nan, np.nan

            pairwise_rows.append({
                "metric_type": metric_type,
                "region": region,
                "cohort_1": c1,
                "cohort_2": c2,
                "n_1": int(len(x1)),
                "n_2": int(len(x2)),
                "mean_1": float(x1.mean()) if len(x1) > 0 else np.nan,
                "mean_2": float(x2.mean()) if len(x2) > 0 else np.nan,
                "diff_mean_1_minus_2": float(x1.mean() - x2.mean()) if len(x1) > 0 and len(x2) > 0 else np.nan,
                "welch_t": t_stat,
                "welch_p": t_p,
                "mannwhitney_U": u_stat,
                "mannwhitney_p": u_p,
                "cohens_d": d,
            })

    omnibus_df = pd.DataFrame(omnibus_rows).sort_values(["metric_type", "region"])
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["metric_type", "region", "cohort_1", "cohort_2"])

    if not pairwise_df.empty:
        pairwise_df["welch_p_fdr"] = np.nan
        pairwise_df["mannwhitney_p_fdr"] = np.nan

        for (metric_type, region), idx in pairwise_df.groupby(["metric_type", "region"]).groups.items():
            idx = list(idx)
            pairwise_df.loc[idx, "welch_p_fdr"] = benjamini_hochberg(pairwise_df.loc[idx, "welch_p"])
            pairwise_df.loc[idx, "mannwhitney_p_fdr"] = benjamini_hochberg(pairwise_df.loc[idx, "mannwhitney_p"])

    return omnibus_df, pairwise_df


# =========================================================
# MAIN
# =========================================================
def main():
    abs_tables = {}
    pct_tables = {}

    for cohort, cfg in DATASETS.items():
        abs_file = cfg["abs_file"]
        pct_file = cfg["pct_file"]

        if not abs_file.exists():
            raise FileNotFoundError(f"Missing absolute file for {cohort}: {abs_file}")
        if not pct_file.exists():
            raise FileNotFoundError(f"Missing percentage file for {cohort}: {pct_file}")

        abs_df = clean_cols(read_table_auto(abs_file))
        pct_df = clean_cols(read_table_auto(pct_file))

        abs_tables[cohort] = abs_df
        pct_tables[cohort] = pct_df

    # -------- build long tables --------
    long_abs = make_long_df(abs_tables, metric_label="absolute")
    long_pct = make_long_df(pct_tables, metric_label="percentage")
    long_all = pd.concat([long_abs, long_pct], axis=0, ignore_index=True)

    long_all_file = RESULTS_DIR / "all_cohorts_long_volumes.csv"
    long_all.to_csv(long_all_file, index=False)

    # -------- within-cohort stats --------
    within_df = compute_within_cohort_stats(long_all)
    within_file = RESULTS_DIR / "within_cohort_descriptive_stats.csv"
    within_df.to_csv(within_file, index=False)

    # -------- between-cohort stats --------
    omnibus_df, pairwise_df = compute_between_cohort_stats(long_all)
    omnibus_file = RESULTS_DIR / "between_cohort_omnibus_stats.csv"
    pairwise_file = RESULTS_DIR / "between_cohort_pairwise_stats.csv"

    omnibus_df.to_csv(omnibus_file, index=False)
    pairwise_df.to_csv(pairwise_file, index=False)

    # -------- figures --------
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric_type in ["absolute", "percentage"]:
        for region in REGION_SPECS.keys():
            sub = long_all[(long_all["metric_type"] == metric_type) & (long_all["region"] == region)].copy()

            if metric_type == "absolute":
                ylabel = "Volume"
            else:
                ylabel = "% of brain volume"

            save_boxplot(
                sub,
                fig_dir / f"{metric_type}_{region}_boxplot.png",
                title=f"{region} ({metric_type}) across cohorts",
                ylabel=ylabel,
            )

            save_histogram_grid(
                sub,
                fig_dir / f"{metric_type}_{region}_histograms.png",
                title=f"{region} ({metric_type}) distributions by cohort",
                xlabel=ylabel,
            )

    # -------- optional region-specific CSVs --------
    region_tables_dir = RESULTS_DIR / "region_tables"
    region_tables_dir.mkdir(parents=True, exist_ok=True)

    for metric_type in ["absolute", "percentage"]:
        for region in REGION_SPECS.keys():
            sub = long_all[(long_all["metric_type"] == metric_type) & (long_all["region"] == region)].copy()
            sub.to_csv(region_tables_dir / f"{metric_type}_{region}_long.csv", index=False)

    # -------- console summary --------
    print("\nSaved:")
    print(long_all_file)
    print(within_file)
    print(omnibus_file)
    print(pairwise_file)
    print(fig_dir)

    print("\nWithin-cohort stats preview:")
    print(within_df.head(12))

    print("\nBetween-cohort omnibus preview:")
    print(omnibus_df.head(8))

    print("\nBetween-cohort pairwise preview:")
    print(pairwise_df.head(12))


if __name__ == "__main__":
    main()