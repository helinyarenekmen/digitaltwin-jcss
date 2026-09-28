"""
paper_figs_exp10.py — Paper-ready figures + significance tests for the
S-vs-P context experiment (exp10_context_peacev3), per meeting notes:

  * policy support DV dropped; DVs = legitimacy + behavioral intent, equal weight
  * "Not reported" / "Other" subgroup levels are excluded from all figures
  * Peace context: significance of Kurdish vs non-Kurdish (and other splits) —
    no barcharts; point estimates + CIs + ANOVA / contrast p-values
  * Security context: ANOVA-style significance for age, ethnicity, ideology,
    education, with one-vs-rest contrasts (does Left diverge? does Kurdish diverge?)
  * One combined figure: all DVs x 4 splits (paired S-P effect forest)

Outputs -> outputs/experiments/exp10_context_peacev3/paper_figs/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp10_context_peacev3"
EXP_DIR = ROOT / "outputs" / "experiments" / EXP_ID
TGSS_PATH = ROOT / "data" / "tgss2024_clean.csv"

OUT_DIR = EXP_DIR / "paper_figs"
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

JSONL_LIKERT = EXP_DIR / f"{EXP_ID}_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / f"{EXP_ID}_C7_gpt4omini_T0_ben_direct_s0.jsonl"

# ---------------------------------------------------------------------------
# DVs (policy support intentionally excluded)
# ---------------------------------------------------------------------------
DV_INFO = {
    "dv_legitimacy": dict(
        label="Movement legitimacy (1–5)", kind="likert"),
    "dv_behavioral_intent": dict(
        label="Would attend the march (share)", kind="binary"),
}
DV_LIST = list(DV_INFO)

COND_LABEL = {"S": "Security", "P": "Peace"}
COND_COLOR = {"S": "#b2182b", "P": "#2166ac"}

# ---------------------------------------------------------------------------
# Subgroups (same recoding as context_subgroup.py; Not reported excluded)
# ---------------------------------------------------------------------------
GENDER_LABEL = {1.0: "Male", 2.0: "Female"}
DEGREE_COLLAPSE = {
    1.0: "Less than high school", 2.0: "Less than high school",
    3.0: "Less than high school", 4.0: "High school",
    5.0: "University or higher", 6.0: "University or higher",
    7.0: "University or higher", 8.0: "University or higher",
}


def age_band(a):
    if pd.isna(a):
        return np.nan
    if a < 30:
        return "18-29"
    if a < 45:
        return "30-44"
    if a < 60:
        return "45-59"
    return "60+"


def politics_band(v):
    if pd.isna(v):
        return np.nan
    if v <= 3:
        return "Left (0-3)"
    if v <= 6:
        return "Center (4-6)"
    return "Right (7-10)"


DIMS = {
    "age_band": dict(
        pretty="Age", order=["18-29", "30-44", "45-59", "60+"], focus=None),
    "education": dict(
        pretty="Education",
        order=["Less than high school", "High school", "University or higher"],
        focus=None),
    "ethnic_origin": dict(
        pretty="Ethnicity", order=["Non-Kurdish", "Kurdish"], focus="Kurdish"),
    "politics": dict(
        pretty="Political orientation",
        order=["Left (0-3)", "Center (4-6)", "Right (7-10)"], focus="Left (0-3)"),
}


def load_experiment() -> pd.DataFrame:
    rows = []
    for fp in (JSONL_LIKERT, JSONL_BINARY):
        for line in fp.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
                continue
            if r["item_id"] not in DV_INFO:
                continue
            y = int(r["predicted_value"])
            if r["item_id"] == "dv_behavioral_intent":
                y = 1 if y == 1 else 0          # 1 = would attend
            rows.append(dict(respondent_id=r["respondent_id"],
                             condition=r["condition"], item_id=r["item_id"], y=y))
    return pd.DataFrame(rows)


def load_subgroups() -> pd.DataFrame:
    df = pd.read_csv(TGSS_PATH, encoding="utf-8",
                     usecols=["id", "age", "gender", "degree",
                              "kurd", "lankurd", "pidleftright"])
    df["respondent_id"] = df["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    df["age_band"] = df["age"].apply(age_band)
    df["education"] = df["degree"].map(DEGREE_COLLAPSE)
    df["ethnic_origin"] = np.where(
        (df["kurd"] == 1.0) | (df["lankurd"] == 1.0), "Kurdish", "Non-Kurdish")
    df["politics"] = df["pidleftright"].apply(politics_band)
    return df[["respondent_id", "age_band", "education",
               "ethnic_origin", "politics"]]


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
def within_condition_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ANOVA per condition x DV x dim, and one-vs-rest Welch contrasts."""
    anova_rows, contrast_rows = [], []
    for cond in ("S", "P"):
        for item in DV_LIST:
            sub = df[(df["condition"] == cond) & (df["item_id"] == item)]
            for dim, meta in DIMS.items():
                groups = [sub.loc[sub[dim] == lv, "y"].to_numpy()
                          for lv in meta["order"]]
                groups = [g for g in groups if len(g) >= 30]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    stat_name, stat_val = "Welch t", t
                else:
                    f, p = stats.f_oneway(*groups)
                    stat_name, stat_val = "F", f
                anova_rows.append(dict(condition=cond, item=item, dim=dim,
                                       test=stat_name, statistic=stat_val,
                                       p=p, stars=p_stars(p)))
                # one-vs-rest focus contrast (Kurdish / Left)
                focus = meta["focus"]
                if focus is not None:
                    a = sub.loc[sub[dim] == focus, "y"].to_numpy()
                    b = sub.loc[(sub[dim] != focus) & sub[dim].notna(), "y"].to_numpy()
                    if len(a) >= 30 and len(b) >= 30:
                        t, p = stats.ttest_ind(a, b, equal_var=False)
                        d = ((a.mean() - b.mean()) /
                             np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
                        contrast_rows.append(dict(
                            condition=cond, item=item, dim=dim, focus=focus,
                            mean_focus=a.mean(), mean_rest=b.mean(),
                            diff=a.mean() - b.mean(), cohens_d=d,
                            welch_t=t, p=p, stars=p_stars(p),
                            n_focus=len(a), n_rest=len(b)))
    return pd.DataFrame(anova_rows), pd.DataFrame(contrast_rows)


def paired_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Within-persona S-P effect per subgroup level (and overall)."""
    rows = []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        wide = sub.pivot_table(index="respondent_id", columns="condition",
                               values="y", aggfunc="first").dropna()
        wide = wide.join(sub.drop_duplicates("respondent_id")
                         .set_index("respondent_id")[list(DIMS)])
        wide["delta"] = wide["S"] - wide["P"]
        t, p = stats.ttest_rel(wide["S"], wide["P"])
        rows.append(dict(item=item, dim="all", level="All respondents",
                         n=len(wide), effect=wide["delta"].mean(),
                         ci=ci95(wide["delta"]), p=p, stars=p_stars(p)))
        for dim, meta in DIMS.items():
            for lv in meta["order"]:
                d = wide.loc[wide[dim] == lv, "delta"]
                if len(d) < 30:
                    continue
                t, p = stats.ttest_rel(
                    wide.loc[wide[dim] == lv, "S"], wide.loc[wide[dim] == lv, "P"])
                rows.append(dict(item=item, dim=dim, level=lv, n=len(d),
                                 effect=d.mean(), ci=ci95(d), p=p,
                                 stars=p_stars(p)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})


def fig_condition_signif(df: pd.DataFrame, anova: pd.DataFrame,
                         contrasts: pd.DataFrame, cond: str, fname: str):
    """Point + 95% CI per subgroup level within one condition; ANOVA p in title,
    focus-contrast p annotated. Rows = DVs, cols = 4 dims."""
    ncol = len(DIMS)
    fig, axes = plt.subplots(len(DV_LIST), ncol,
                             figsize=(2.9 * ncol, 2.6 * len(DV_LIST)),
                             sharey="row")
    for i, item in enumerate(DV_LIST):
        sub = df[(df["condition"] == cond) & (df["item_id"] == item)]
        for j, (dim, meta) in enumerate(DIMS.items()):
            ax = axes[i, j]
            order = meta["order"]
            means = [sub.loc[sub[dim] == lv, "y"].mean() for lv in order]
            cis = [ci95(sub.loc[sub[dim] == lv, "y"].to_numpy()) for lv in order]
            xs = np.arange(len(order))
            ax.errorbar(xs, means, yerr=cis, fmt="o", ms=4.5, lw=1.4,
                        capsize=3, color=COND_COLOR[cond])
            arow = anova[(anova["condition"] == cond) & (anova["item"] == item)
                         & (anova["dim"] == dim)]
            title = meta["pretty"]
            if len(arow):
                r = arow.iloc[0]
                stat = (f"t = {r['statistic']:.2f}" if r["test"] == "Welch t"
                        else f"F = {r['statistic']:.2f}")
                title += f"\n{stat}, p {'< 0.001' if r['p'] < 0.001 else f'= {r.p:.3f}'}"
            ax.set_title(title, fontsize=8.5)
            crow = contrasts[(contrasts["condition"] == cond)
                             & (contrasts["item"] == item)
                             & (contrasts["dim"] == dim)]
            if len(crow):
                r = crow.iloc[0]
                k = order.index(r["focus"])
                ax.annotate(r["stars"], (xs[k], means[k]),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9, fontweight="bold")
            ax.set_xticks(xs)
            ax.set_xticklabels([o.replace(" (", "\n(").replace("Less than high school", "< High\nschool").replace("University or higher", "University\nor higher").replace("High school", "High\nschool")
                                for o in order], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(DV_INFO[item]["label"], fontsize=8.5)
    fig.suptitle(f"{COND_LABEL[cond]} context — subgroup means, "
                 f"95% CIs, and significance tests", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_effect_forest(pe: pd.DataFrame, fname: str):
    """Single figure: paired S-P context effect for every DV x 4 splits."""
    fig, axes = plt.subplots(1, len(DV_LIST), figsize=(4.6 * len(DV_LIST), 5.6),
                             sharey=True)
    ylabels_done = False
    for k, item in enumerate(DV_LIST):
        ax = axes[k]
        rows, labels, ypos = [], [], []
        y = 0
        for dim, meta in DIMS.items():
            block = pe[(pe["item"] == item) & (pe["dim"] == dim)]
            if not len(block):
                continue
            esc = meta["pretty"].replace(" ", "\\ ")
            labels.append(f"$\\bf{{{esc}}}$")
            ypos.append(y)
            rows.append(None)
            y -= 1
            for lv in meta["order"]:
                r = block[block["level"] == lv]
                if not len(r):
                    continue
                rows.append(r.iloc[0])
                labels.append(f"  {lv}")
                ypos.append(y)
                y -= 1
            y -= 0.4
        overall = pe[(pe["item"] == item) & (pe["dim"] == "all")].iloc[0]
        for yy, r in zip(ypos, rows):
            if r is None:
                continue
            sig = r["p"] < 0.05
            ax.errorbar(r["effect"], yy, xerr=r["ci"], fmt="s",
                        ms=4.5, lw=1.4, capsize=3,
                        color="#1a1a1a" if sig else "#9a9a9a")
        ax.axvline(0, color="#888", lw=0.8, ls=":")
        ax.axvline(overall["effect"], color="#b2182b", lw=1.0, ls="--",
                   label=f"Overall effect ({overall['effect']:+.2f})")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(DV_INFO[item]["label"], fontsize=9.5)
        ax.set_xlabel("Security − Peace (within-persona, 95% CI)")
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.suptitle("Context effect (Security − Peace) by subgroup", y=0.98,
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_main_summary(df: pd.DataFrame, fname: str):
    """Paper version of fig08: overall condition means per DV, 2 panels."""
    fig, axes = plt.subplots(1, len(DV_LIST), figsize=(3.1 * len(DV_LIST), 3.0))
    for k, item in enumerate(DV_LIST):
        ax = axes[k]
        for x, cond in enumerate(("P", "S")):
            v = df[(df["condition"] == cond) & (df["item_id"] == item)]["y"]
            ax.errorbar(x, v.mean(), yerr=ci95(v.to_numpy()), fmt="o", ms=6,
                        capsize=4, lw=1.6, color=COND_COLOR[cond])
        wide = (df[df["item_id"] == item]
                .pivot_table(index="respondent_id", columns="condition",
                             values="y", aggfunc="first").dropna())
        t, p = stats.ttest_rel(wide["S"], wide["P"])
        dz = (wide["S"] - wide["P"]).mean() / (wide["S"] - wide["P"]).std(ddof=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([COND_LABEL["P"], COND_LABEL["S"]])
        ax.set_title(f"{DV_INFO[item]['label']}\n"
                     f"paired t = {t:.1f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}, "
                     f"$d_z$ = {dz:.2f}", fontsize=9)
        ax.set_xlim(-0.5, 1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    df = load_experiment().merge(load_subgroups(), on="respondent_id", how="left")
    print(f"records: {len(df)}, respondents: {df.respondent_id.nunique()}")

    anova, contrasts = within_condition_tests(df)
    pe = paired_effects(df)
    anova.to_csv(CSV_DIR / "anova_within_condition.csv", index=False)
    contrasts.to_csv(CSV_DIR / "focus_contrasts.csv", index=False)
    pe.to_csv(CSV_DIR / "paired_effects_by_subgroup.csv", index=False)

    fig_condition_signif(df, anova, contrasts, "P", "fig_peace_subgroup_signif")
    fig_condition_signif(df, anova, contrasts, "S", "fig_security_subgroup_signif")
    fig_effect_forest(pe, "fig_context_effect_forest")
    fig_main_summary(df, "fig_context_main")

    print("\n== ANOVA (within condition) ==")
    print(anova.to_string(index=False))
    print("\n== Focus contrasts (Kurdish / Left one-vs-rest) ==")
    print(contrasts.to_string(index=False))
    print("\n== Overall paired effects ==")
    print(pe[pe["dim"] == "all"].to_string(index=False))
    print(f"\nfigures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
