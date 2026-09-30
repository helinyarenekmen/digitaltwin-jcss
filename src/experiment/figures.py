"""
Figure generators for paper Figures 2, 5, 6 (distributions), 3, 4
(forests), 7, 8 (subgroup interactions), 9, 10 (NUTS-1 maps).

Each generator reads only from outputs/parsed/ and outputs/results/, plus
data/derived/tgss2024_clean.csv and data/derived/geo/*.geojson for the maps.
No API calls.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
PARSED  = ROOT / "outputs" / "parsed"
RESULTS = ROOT / "outputs" / "results"
FIGDIR  = ROOT / "figures"

# --- Shared style ---
rcParams["font.family"]  = "DejaVu Sans"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42
INK, INK_SOFT, INK_MUTED = "#1F2937", "#4B5563", "#6B7280"
GRID = "#E5E7EB"
POINT = "#1F3B6B"
COL_THREAT = "#3B76B0"
COL_OPPORT = "#D97441"


def _save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _long():
    df = pd.read_csv(PARSED / "all_items_long.csv")
    return df[df["parse_status"] == "ok"].copy()


def _tgss_bands():
    tgss = pd.read_csv(ROOT / "data" / "derived" / "tgss2024_clean.csv")
    tgss["respondent_id"] = tgss["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")

    def _age(a):
        if pd.isna(a): return None
        return "18-29" if a<30 else "30-44" if a<45 else "45-59" if a<60 else "60+"
    def _edu(v):
        if pd.isna(v): return None
        return "Less than high school" if v<=3 else ("High school" if v==4 else "University or higher")
    def _pol(v):
        if pd.isna(v): return None
        return "Left" if v<=3 else ("Center" if v<=6 else "Right")
    def _eth(r):
        return "Kurdish" if (r.get("kurd")==1.0 or r.get("lankurd")==1.0) else "Non-Kurdish"
    tgss["age_band"]  = tgss["age"].apply(_age)
    tgss["education"] = tgss["degree"].apply(_edu)
    tgss["ethnicity"] = tgss.apply(_eth, axis=1)
    tgss["politics"]  = tgss["pidleftright"].apply(_pol)
    return tgss


# ---------------------------------------------------------------------------
# Fig 2 — effect summary (2 panels, horizontal error bars, dashed 0-line)
# ---------------------------------------------------------------------------
def fig02_effect_summary():
    df = _long()

    def _leg():
        d = df[df["item_id"] == "dv_legitimacy"]
        w = d.pivot_table(index="respondent_id", columns="condition",
                          values="predicted_value", aggfunc="first").dropna()
        diff = (w["S"] - w["P"]).astype(float).values
        n = len(diff); m = float(diff.mean()); s = float(diff.std(ddof=1)/n**0.5)
        return n, m, m-1.96*s, m+1.96*s, m/diff.std(ddof=1)

    def _beh():
        d = df[df["item_id"] == "dv_behavioral_intent"]
        w = d.pivot_table(index="respondent_id", columns="condition",
                          values="predicted_value", aggfunc="first").dropna()
        S = (w["S"] == 1).astype(float).values
        P = (w["P"] == 1).astype(float).values
        diff = (S - P) * 100
        n = len(diff); m = float(diff.mean()); s = float(diff.std(ddof=1)/n**0.5)
        return n, m, m-1.96*s, m+1.96*s

    n_leg, d_leg, lo_leg, hi_leg, dz = _leg()
    n_beh, d_beh, lo_beh, hi_beh = _beh()

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.5, 5.2))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14, hspace=0.95)

    def _panel(ax, d, lo, hi, xlabel, xlim, tick, letter, title, s1, s2, n):
        ax.errorbar(d, 0, xerr=[[d-lo],[hi-d]], fmt="o", color=POINT, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.3,
                    capsize=7, capthick=1.5, elinewidth=1.8, zorder=3)
        ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks([]); ax.set_ylim(-1,1); ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.set_xlabel(xlabel, fontsize=11, color=INK_SOFT, labelpad=6)
        for s in ("top","right","left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(colors=INK_SOFT, labelsize=10.5, width=0.7, length=3)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
        ax.text(0.0, 1.16, f"{letter}  {title}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12.5, weight="bold", color=INK)
        ax.text(1.0, 1.16, f"{s1}      {s2}      n = {n:,}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color=INK)

    _panel(axA, d_leg, lo_leg, hi_leg,
           "Security − Peace  (Likert points, 1–5)", (-0.75, 0.35), 0.20,
           "A", "Perceived legitimacy",
           f"Δ = {d_leg:.2f}  95% CI [{lo_leg:.2f}, {hi_leg:.2f}]",
           f"p < .001  ·  $d_{{z}}$ = {dz:.2f}", n_leg)
    _panel(axB, d_beh, lo_beh, hi_beh,
           "Security − Peace  (percentage points)", (-22, 8), 5,
           "B", "Behavioral intent",
           f"Δ = {d_beh:.2f} pp  95% CI [{lo_beh:.2f}, {hi_beh:.2f}] pp",
           "McNemar p < .001", n_beh)
    _save(fig, "Fig02_effect_summary")


# ---------------------------------------------------------------------------
# Fig 5 — legitimacy distribution (grouped bar chart, 5 categories)
# ---------------------------------------------------------------------------
def fig05_legitimacy_distribution():
    df = _long()
    d = df[df["item_id"] == "dv_legitimacy"].copy()
    d["predicted_value"] = d["predicted_value"].astype(int)
    cats = np.arange(1, 6)
    labels = ["Strongly disagree", "Disagree", "Neither", "Agree", "Strongly agree"]
    pS = np.array([(d[d.condition=="S"]["predicted_value"]==c).mean()*100 for c in cats])
    pP = np.array([(d[d.condition=="P"]["predicted_value"]==c).mean()*100 for c in cats])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14)
    x = np.arange(5); w = 0.38
    ax.bar(x-w/2, pS, width=w, color=COL_THREAT, edgecolor="none", label="Threat", zorder=3)
    ax.bar(x+w/2, pP, width=w, color=COL_OPPORT, edgecolor="none", label="Opportunity", zorder=3)
    top = max(pS.max(), pP.max()) * 1.15
    for i in range(5):
        ax.text(i-w/2, pS[i]+top*0.015, f"{pS[i]:.1f}%", ha="center", va="bottom",
                fontsize=10, color=COL_THREAT, weight="semibold")
        ax.text(i+w/2, pP[i]+top*0.015, f"{pP[i]:.1f}%", ha="center", va="bottom",
                fontsize=10, color=COL_OPPORT, weight="semibold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_ylabel("% of synthetic respondents", fontsize=11.5, color=INK_SOFT, labelpad=6)
    ax.set_ylim(0, top)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=11, width=0.7, length=3, pad=4)
    ax.tick_params(axis="y", left=False, labelleft=False); ax.grid(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=11, frameon=False, handlelength=1.4, columnspacing=2.5)
    _save(fig, "Fig05_legitimacy_distribution")


# ---------------------------------------------------------------------------
# Fig 6 — behavioral distribution (2 categories)
# ---------------------------------------------------------------------------
def fig06_behavioral_distribution():
    df = _long()
    d = df[df["item_id"] == "dv_behavioral_intent"].copy()
    d["predicted_value"] = d["predicted_value"].astype(int)
    # 1 = would attend, 2 = would not attend  →  show as (Would not attend, Would attend)
    labels = ["Would not attend", "Would attend"]
    pS = np.array([(d[d.condition=="S"]["predicted_value"]==c).mean()*100 for c in (2, 1)])
    pP = np.array([(d[d.condition=="P"]["predicted_value"]==c).mean()*100 for c in (2, 1)])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16)
    x = np.arange(2); w = 0.38
    ax.bar(x-w/2, pS, width=w, color=COL_THREAT, edgecolor="none", label="Threat", zorder=3)
    ax.bar(x+w/2, pP, width=w, color=COL_OPPORT, edgecolor="none", label="Opportunity", zorder=3)
    top = max(pS.max(), pP.max()) * 1.15
    for i in range(2):
        ax.text(i-w/2, pS[i]+top*0.015, f"{pS[i]:.1f}%", ha="center", va="bottom",
                fontsize=10, color=COL_THREAT, weight="semibold")
        ax.text(i+w/2, pP[i]+top*0.015, f"{pP[i]:.1f}%", ha="center", va="bottom",
                fontsize=10, color=COL_OPPORT, weight="semibold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_xlabel("Behavioral intent", fontsize=11.5, color=INK_SOFT, labelpad=8)
    ax.set_ylabel("% of synthetic respondents", fontsize=11.5, color=INK_SOFT, labelpad=6)
    ax.set_ylim(0, top)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=11, width=0.7, length=3, pad=4)
    ax.tick_params(axis="y", left=False, labelleft=False); ax.grid(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=11, frameon=False, handlelength=1.4, columnspacing=2.5)
    _save(fig, "Fig06_behavioral_distribution")


# ---------------------------------------------------------------------------
# Fig 3 — Between-group forest within each framing condition (Peace + Security)
# Fig 4 — Subgroup context-effect forest (Peace − Security)
# Reuse the archived contrasts / subgroup_means CSVs for the data.
# ---------------------------------------------------------------------------
def fig04_subgroup_context_forest():
    """Peace − Security paired contrast for each subgroup dimension × DV.
    2 columns (legitimacy, behavioral) × 4 rows (age, education, ethnicity, politics).
    """
    contrasts = pd.read_csv(RESULTS / "contrasts.csv")

    DIM_ORDER = {
        "age_band":      ["18-29", "30-44", "45-59", "60+"],
        "education":     ["Less than high school", "High school", "University or higher"],
        "ethnic_origin": ["Non-Kurdish", "Kurdish"],
        "politics":      ["Left (0-3)", "Center (4-6)", "Right (7-10)"],
    }
    LABEL_MAP = {"Left (0-3)": "Left", "Center (4-6)": "Center", "Right (7-10)": "Right"}
    DIM_PRETTY = {"age_band": "Age", "education": "Education",
                   "ethnic_origin": "Ethnic origin", "politics": "Political orientation"}
    DV_META = {
        "dv_legitimacy": {"title": "Movement legitimacy",
                          "xlabel": "Opportunity − Threat  (Likert points)", "scale": 1.0},
        "dv_behavioral_intent": {"title": "Behavioral intent",
                                  "xlabel": "Opportunity − Threat  (percentage points)", "scale": 100.0},
    }

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.87, bottom=0.05, hspace=0.75, wspace=0.22)
    xlim = {0: [-0.5, 0.7], 1: [-30, 30]}
    for row_idx, (dim, levels) in enumerate(DIM_ORDER.items()):
        for col_idx, (item, meta) in enumerate(DV_META.items()):
            ax = axes[row_idx, col_idx]
            sub = contrasts[(contrasts["dim"] == dim) & (contrasts["item"] == item)
                            & (contrasts["level"].isin(levels))].copy()
            sub["order"] = sub["level"].map({l: i for i, l in enumerate(levels)})
            sub = sub.sort_values("order")
            ys = np.arange(len(sub))
            v = sub["diff_P_minus_S"].values * meta["scale"]
            lo = sub["ci_lo"].values * meta["scale"]
            hi = sub["ci_hi"].values * meta["scale"]
            ax.errorbar(v, ys, xerr=[v-lo, hi-v], fmt="o", color=POINT, markersize=7.5,
                        markeredgecolor="white", capsize=4, capthick=1.4, elinewidth=1.5, zorder=3)
            ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
            ax.set_yticks(ys)
            if col_idx == 0:
                labels = [LABEL_MAP.get(l, l) for l in sub["level"]]
                ax.set_yticklabels(labels, fontsize=10, color=INK)
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            if row_idx == 0:
                ax.set_title(meta["title"], fontsize=11, weight="semibold", color=INK,
                             loc="left", pad=8)
            if row_idx == len(DIM_ORDER) - 1:
                ax.set_xlabel(meta["xlabel"], fontsize=10, color=INK_SOFT, labelpad=6)
            else:
                ax.set_xticklabels([])
            for s in ("top","right","left"): ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
            ax.tick_params(axis="x", colors=INK_SOFT, labelsize=9, width=0.7, length=3)
            ax.tick_params(axis="y", left=False); ax.grid(False)
            xlim[col_idx][0] = min(xlim[col_idx][0], float(lo.min())*1.15 - 0.05)
            xlim[col_idx][1] = max(xlim[col_idx][1], float(hi.max())*1.15 + 0.05)
    for r in range(4):
        for c in range(2):
            axes[r, c].set_xlim(*xlim[c])
    for row_idx, dim in enumerate(DIM_ORDER):
        pos = axes[row_idx, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.032, DIM_PRETTY[dim], ha="left", va="bottom",
                 fontsize=12, weight="bold", color=INK)
    fig.suptitle("Political-context effects across subgroups",
                 fontsize=13, weight="bold", color=INK, x=0.15, ha="left", y=0.965)
    _save(fig, "Fig04_subgroup_context_forest")


def fig03_between_group_forest():
    """Between-group contrasts within each framing condition (Threat and
    Opportunity), 4 rows × 2 columns. Uses paired subgroup contrasts against a
    reference (18-29 for age, University+ for education, etc.)."""
    df = _long()
    tgss = _tgss_bands()
    meta = tgss[["respondent_id", "age_band", "education", "ethnicity", "politics"]]
    df = df.merge(meta, on="respondent_id", how="inner")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    CONTRASTS = {
        "age_band":  [("18-29","30-44"),("18-29","45-59"),("18-29","60+")],
        "education": [("University or higher","High school"),
                       ("University or higher","Less than high school")],
        "ethnicity": [("Kurdish","Non-Kurdish")],
        "politics":  [("Left","Center"),("Left","Right")],
    }
    LABEL_MAP = {"University or higher":"University+","Less than high school":"Less than HS",
                  "High school":"High school"}
    def _fmt(t,r): return f"{LABEL_MAP.get(t,t)} − {LABEL_MAP.get(r,r)}"

    def tukey(sub, dim, pairs):
        levels = sorted({l for pair in pairs for l in pair})
        dat = sub[sub[dim].isin(levels)]
        if dat[dim].nunique() < 2: return []
        tuk = pairwise_tukeyhsd(dat["predicted_value"].astype(float).values, dat[dim].values)
        res = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
        out = []
        for t, r in pairs:
            row = res[(res["group1"]==r) & (res["group2"]==t)]
            flip = False
            if row.empty:
                row = res[(res["group1"]==t) & (res["group2"]==r)]; flip = True
            if row.empty: continue
            row = row.iloc[0]
            d = float(row["meandiff"]); lo = float(row["lower"]); hi = float(row["upper"])
            if flip: d = -d; lo, hi = -hi, -lo
            out.append({"pair": _fmt(t, r), "diff": d, "ci_lo": lo, "ci_hi": hi})
        return out

    def prop(sub, dim, pairs):
        out = []; ps = {}
        for lvl, g in sub.groupby(dim):
            v = (g["predicted_value"]==1).astype(float).values
            ps[lvl] = (v.mean(), len(v))
        for t, r in pairs:
            if t not in ps or r not in ps: continue
            p1, n1 = ps[t]; p2, n2 = ps[r]
            d = (p1-p2)*100
            se = ((p1*(1-p1)/n1 + p2*(1-p2)/n2)**0.5)*100
            out.append({"pair": _fmt(t,r), "diff": d, "ci_lo": d-1.96*se, "ci_hi": d+1.96*se})
        return out

    results = {"P": {"leg": {}, "beh": {}}, "S": {"leg": {}, "beh": {}}}
    for cond in ("P","S"):
        s = df[df["condition"]==cond]
        leg = s[s["item_id"]=="dv_legitimacy"]
        beh = s[s["item_id"]=="dv_behavioral_intent"]
        for dim, pairs in CONTRASTS.items():
            results[cond]["leg"][dim] = tukey(leg, dim, pairs)
            results[cond]["beh"][dim] = prop(beh, dim, pairs)

    DIM_ORDER = list(CONTRASTS)
    DIM_PRETTY = {"age_band":"Age","education":"Education","ethnicity":"Ethnic origin","politics":"Political orientation"}
    DV_META = {0: {"title":"Movement legitimacy","xlabel":"Within-condition contrast  (Likert points, Tukey-adjusted)"},
                1: {"title":"Behavioral intent","xlabel":"Within-condition contrast  (percentage points, binomial marginal prob.)"}}
    row_heights = [len(CONTRASTS[d]) for d in DIM_ORDER]

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(4, 2, height_ratios=row_heights, left=0.18, right=0.94,
                          top=0.86, bottom=0.09, hspace=0.55, wspace=0.34)
    axes = np.empty((4, 2), dtype=object)
    for r in range(4):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])
    xlim = {0: [-0.6, 1.2], 1: [-30, 60]}
    OFFSET = 0.18
    for row_idx, dim in enumerate(DIM_ORDER):
        for col_idx, dv_key in enumerate(("leg","beh")):
            ax = axes[row_idx, col_idx]
            rs = results["S"][dv_key][dim]; rp = results["P"][dv_key][dim]
            ys = np.arange(len(rp))
            for off, recs, colour, lbl in [(-OFFSET, rs, COL_THREAT, "Threat (security)"),
                                             (+OFFSET, rp, COL_OPPORT, "Opportunity (peace)")]:
                v = np.array([r["diff"] for r in recs])
                lo = np.array([r["ci_lo"] for r in recs])
                hi = np.array([r["ci_hi"] for r in recs])
                ax.errorbar(v, ys+off, xerr=[v-lo, hi-v], fmt="o", color=colour, markersize=6.5,
                            markeredgecolor="white", capsize=3.5, capthick=1.2, elinewidth=1.3,
                            zorder=3, label=lbl if (row_idx==0 and col_idx==0) else None)
            ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
            ax.set_yticks(ys)
            if col_idx == 0:
                ax.set_yticklabels([r["pair"] for r in rp], fontsize=10, color=INK)
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            if row_idx == len(DIM_ORDER)-1:
                ax.set_xlabel(DV_META[col_idx]["xlabel"], fontsize=10, color=INK_SOFT, labelpad=6)
            else:
                ax.set_xticklabels([])
            for s in ("top","right"): ax.spines[s].set_visible(False)
            for s in ("left","bottom"):
                ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.7)
            ax.tick_params(colors=INK_SOFT, labelsize=9, width=0.7, length=3)
            ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
            all_lo = min([r["ci_lo"] for r in rp+rs])
            all_hi = max([r["ci_hi"] for r in rp+rs])
            xlim[col_idx][0] = min(xlim[col_idx][0], float(all_lo)*1.15 - 0.05)
            xlim[col_idx][1] = max(xlim[col_idx][1], float(all_hi)*1.15 + 0.05)
    for r in range(4):
        for c in range(2):
            axes[r, c].set_xlim(*xlim[c])
    COL_TITLE_Y = 0.910
    for col_idx in range(2):
        pos = axes[0, col_idx].get_position()
        fig.text(pos.x0, COL_TITLE_Y, DV_META[col_idx]["title"], ha="left", va="bottom",
                 fontsize=12, weight="semibold", color=INK)
    for row_idx, dim in enumerate(DIM_ORDER):
        pos = axes[row_idx, 0].get_position()
        y = min(pos.y1 + 0.012, COL_TITLE_Y - 0.028) if row_idx == 0 else pos.y1 + 0.012
        fig.text(pos.x0, y, DIM_PRETTY[dim], ha="left", va="bottom",
                 fontsize=11, weight="bold", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.985),
               ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Between-group contrasts within each framing condition",
                 fontsize=14, weight="bold", color=INK, x=0.18, ha="left", y=0.975)
    _save(fig, "Fig03_between_group_forest")


# ---------------------------------------------------------------------------
# Fig 7, 8 — subgroup × context interaction
# ---------------------------------------------------------------------------
def _fig_interaction(item: str, out_stem: str):
    means = pd.read_csv(RESULTS / "subgroup_means.csv")
    DIM_ORDER = {"age_band": ["18-29","30-44","45-59","60+"],
                  "education": ["Less than high school","High school","University or higher"],
                  "ethnic_origin": ["Non-Kurdish","Kurdish"],
                  "politics": ["Left (0-3)","Center (4-6)","Right (7-10)"]}
    LEVEL_DISPLAY = {"University or higher":"University+","Less than high school":"Less than HS",
                      "Left (0-3)":"Left","Center (4-6)":"Center","Right (7-10)":"Right"}
    DIM_PRETTY = {"age_band":"Age","education":"Education","ethnic_origin":"Ethnic origin","politics":"Political orientation"}
    scale = 100.0 if item == "dv_behavioral_intent" else 1.0
    ylabel = "% would attend" if item == "dv_behavioral_intent" else "Mean response (1–5)"
    title = "Subgroup × framing interaction — " + (
        "Behavioral intent" if item == "dv_behavioral_intent" else "Movement legitimacy")

    COL_LEVEL = ["#1F3B6B", "#B03A3A", "#4E7D3E", "#6B4E9C"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.10, hspace=0.55, wspace=0.25)
    all_vals = []
    for dim in DIM_ORDER:
        sub = means[(means["item"]==item) & (means["dim"]==dim) & (means["level"].isin(DIM_ORDER[dim]))]
        all_vals.extend((sub["mean"]*scale).tolist())
    ymin = min(all_vals) - (max(all_vals)-min(all_vals))*0.10
    ymax = max(all_vals) + (max(all_vals)-min(all_vals))*0.20
    for i, dim in enumerate(DIM_ORDER):
        r, c = divmod(i, 2); ax = axes[r, c]
        levels = DIM_ORDER[dim]
        for j, lvl in enumerate(levels):
            row = means[(means["item"]==item)&(means["dim"]==dim)&(means["level"]==lvl)]
            if row.empty: continue
            yS = float(row[row["condition"]=="S"]["mean"].iloc[0])*scale
            yP = float(row[row["condition"]=="P"]["mean"].iloc[0])*scale
            seS = float(row[row["condition"]=="S"]["se"].iloc[0])*scale
            seP = float(row[row["condition"]=="P"]["se"].iloc[0])*scale
            nS = int(row[row["condition"]=="S"]["n"].iloc[0])
            ax.errorbar([0, 1], [yS, yP], yerr=[1.96*seS, 1.96*seP],
                        fmt="-o", color=COL_LEVEL[j], markersize=6.5, markeredgecolor="white",
                        markeredgewidth=1.0, linewidth=1.8, capsize=3.5, capthick=1.1,
                        elinewidth=1.1, zorder=3, label=f"{LEVEL_DISPLAY.get(lvl, lvl)}  (n={nS:,})")
        ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1])
        ax.set_xticklabels(["Threat", "Opportunity"], fontsize=10.5, color=INK)
        ax.set_ylim(ymin, ymax)
        ax.set_title(DIM_PRETTY[dim], fontsize=11.5, weight="bold", color=INK, loc="left", pad=6)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=10.5, color=INK_SOFT, labelpad=6)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        for s in ("left","bottom"):
            ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.7)
        ax.tick_params(colors=INK_SOFT, labelsize=10, width=0.7, length=3)
        ax.grid(axis="y", color="#EEF0F3", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9, frameon=False, handlelength=1.6)
    fig.suptitle(title, fontsize=13.5, weight="bold", color=INK, x=0.07, ha="left", y=0.97)
    _save(fig, out_stem)


def fig07_subgroup_interaction_behavioral():
    _fig_interaction("dv_behavioral_intent", "Fig07_subgroup_interaction_behavioral")


def fig08_subgroup_interaction_legitimacy():
    _fig_interaction("dv_legitimacy", "Fig08_subgroup_interaction_legitimacy")


# ---------------------------------------------------------------------------
# Fig 9, 10 — NUTS-1 choropleth maps
# ---------------------------------------------------------------------------
def _regional_map(outcome: str, xlabel: str, out_stem: str):
    import geopandas as gpd
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator

    NUTS1_LABEL = {1:"İstanbul", 2:"Western Marmara", 3:"Aegean", 4:"Eastern Marmara",
                    5:"Western Anatolia", 6:"Mediterranean", 7:"Central Anatolia",
                    8:"Western Black Sea", 9:"Eastern Black Sea", 10:"Northeast Anatolia",
                    11:"East-Central Anatolia", 12:"Southeast Anatolia"}
    NUTS1_ID = {i: f"TR{c}" for i, c in enumerate(
        ["1","2","3","4","5","6","7","8","9","A","B","C"], start=1)}

    df = _long()
    d = df[df["item_id"] == outcome].copy()
    d["predicted_value"] = d["predicted_value"].astype(int)
    tgss = pd.read_csv(ROOT / "data" / "derived" / "tgss2024_clean.csv",
                        usecols=["id", "nuts1"])
    tgss["respondent_id"] = tgss["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    tgss["nuts_id"] = tgss["nuts1"].astype(int).map(NUTS1_ID)
    tgss["region"] = tgss["nuts1"].astype(int).map(NUTS1_LABEL)
    d = d.merge(tgss[["respondent_id", "region", "nuts_id"]], on="respondent_id", how="inner")

    wide = d.pivot_table(index=["respondent_id","region","nuts_id"], columns="condition",
                          values="predicted_value", aggfunc="first").dropna(subset=["S","P"]).reset_index()
    if outcome == "dv_behavioral_intent":
        wide["effect"] = ((wide["P"]==1).astype(int) - (wide["S"]==1).astype(int)) * 100.0
        tick_step = 2
    else:
        wide["effect"] = wide["P"] - wide["S"]
        tick_step = 0.05
    per_region = wide.groupby(["region","nuts_id"])["effect"].mean().reset_index()

    gdf = gpd.read_file(ROOT / "data" / "derived" / "geo" / "nuts1_eu_2021.geojson")
    gdf = gdf[gdf["CNTR_CODE"] == "TR"][["NUTS_ID", "geometry"]].merge(
        per_region, left_on="NUTS_ID", right_on="nuts_id", how="left")
    vmin, vmax = float(gdf["effect"].min()), float(gdf["effect"].max())
    cmap = cm.get_cmap("Oranges" if vmin >= 0 else "RdBu_r")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    fig.subplots_adjust(left=0.02, right=0.94, top=0.99, bottom=0.02)
    gdf.plot(column="effect", cmap=cmap, norm=norm, ax=ax, edgecolor="white",
             linewidth=0.5, missing_kwds={"color":"#E5E7EB","edgecolor":"white"})
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.015, aspect=22)
    cbar.set_label(xlabel, fontsize=10.5, color=INK_SOFT, labelpad=8)
    cbar.ax.tick_params(labelsize=9.5, colors=INK_SOFT)
    cbar.outline.set_edgecolor(INK_MUTED); cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_axis_off()
    _save(fig, out_stem)


def fig09_regional_legitimacy_map():
    _regional_map("dv_legitimacy",
                   "Opportunity − Threat (Likert points)",
                   "Fig09_regional_legitimacy_map")


def fig10_regional_behavioral_map():
    _regional_map("dv_behavioral_intent",
                   "Opportunity − Threat (percentage points)",
                   "Fig10_regional_behavioral_map")
