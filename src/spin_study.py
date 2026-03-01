import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, pearsonr
import statsmodels.formula.api as smf

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    HRFlowable, Table, TableStyle
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors


# =========================
# SETTINGS (STRICT 26, CACHE-ONLY)
# =========================
YEAR = 2024
PITCH_TYPE = "FF"
MAX_WIND_MPH = 5
MIN_PITCHES_PER_BUCKET = 75
N_PITCHERS = 26

CACHE_FILE = f"clean_{PITCH_TYPE}_{YEAR}_JunJul_ET_wind{MAX_WIND_MPH}_rho.parquet"

OUT_FIG = "FIG_spin_ineff_diff_sorted.png"
OUT_CSV = f"summary_{YEAR}_JunJul_{PITCH_TYPE}_ET_wind{MAX_WIND_MPH}_strict{N_PITCHERS}.csv"
OUT_PDF = f"Spin_Report_{YEAR}_JunJul_{PITCH_TYPE}_ET_wind{MAX_WIND_MPH}_ResearchBrief.pdf"


# =========================
# HELPERS
# =========================
def short_name(name: str) -> str:
    # "Last, First" -> "Last, F"
    if isinstance(name, str) and "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return f"{last}, {first[:1]}"
    return str(name)

def select_pitchers_strict(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["pitcher", "player_name", "time_bucket"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    eligible = counts[
        (counts.get("pre_2pm", 0) >= MIN_PITCHES_PER_BUCKET) &
        (counts.get("post_6pm", 0) >= MIN_PITCHES_PER_BUCKET)
    ].copy()

    eligible["total"] = eligible["pre_2pm"] + eligible["post_6pm"]
    eligible = eligible.sort_values("total", ascending=False)

    print("\nEligible pitchers meeting minimums:", len(eligible))
    print(eligible.head(10).to_string(index=False))

    n_use = min(N_PITCHERS, len(eligible))
    top = eligible.head(n_use)[["pitcher", "player_name"]]
    return df.merge(top, on=["pitcher", "player_name"], how="inner")

def build_pitcher_summary(df_sel: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df_sel.groupby(["player_name", "time_bucket"])
              .agg(
                  spin_rate_avg=("release_spin_rate", "mean"),
                  spin_eff_avg=("spin_eff", "mean"),
                  spin_ineff_avg=("spin_ineff", "mean"),
                  velo_avg=("release_speed", "mean"),
                  n=("release_speed", "size"),
              )
              .reset_index()
    )

    pivot = grp.pivot(index="player_name", columns="time_bucket")
    pivot.columns = ["_".join(col) for col in pivot.columns]
    pivot = pivot.reset_index()

    # Require both buckets
    pivot = pivot.dropna(subset=["spin_ineff_avg_pre_2pm", "spin_ineff_avg_post_6pm"]).copy()

    # Diffs (post - pre)
    pivot["spin_ineff_diff"] = pivot["spin_ineff_avg_post_6pm"] - pivot["spin_ineff_avg_pre_2pm"]
    pivot["velo_diff"] = pivot["velo_avg_post_6pm"] - pivot["velo_avg_pre_2pm"]
    pivot["spin_rate_diff"] = pivot["spin_rate_avg_post_6pm"] - pivot["spin_rate_avg_pre_2pm"]
    pivot["spin_eff_diff"] = pivot["spin_eff_avg_post_6pm"] - pivot["spin_eff_avg_pre_2pm"]

    return pivot

def make_sorted_bar(pivot: pd.DataFrame) -> str:
    p = pivot.sort_values("spin_ineff_diff", ascending=True).copy()
    labels = [short_name(x) for x in p["player_name"].tolist()]

    plt.figure(figsize=(10.5, 4.6))
    plt.bar(labels, p["spin_ineff_diff"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("Δ Spin ineff (post − pre)")
    plt.title("Spin inefficiency change by pitcher (Post 6pm − Pre 2pm) | June–July | FF | Wind≤5mph | ET")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=350)
    plt.close()
    return OUT_FIG

def mixedlm_pitch_level(df_sel: pd.DataFrame):
    dfm = df_sel.copy()

    dfm["release_speed"] = pd.to_numeric(dfm["release_speed"], errors="coerce")
    dfm["spin_ineff"] = pd.to_numeric(dfm["spin_ineff"], errors="coerce")
    dfm = dfm.dropna(subset=["release_speed", "spin_ineff", "pitcher", "time_bucket"]).copy()

    dfm["is_post6"] = (dfm["time_bucket"] == "post_6pm").astype(int)

    sd = dfm["release_speed"].std(ddof=1)
    if sd == 0 or np.isnan(sd):
        dfm["velo_z"] = 0.0
    else:
        dfm["velo_z"] = (dfm["release_speed"] - dfm["release_speed"].mean()) / sd

    formula = "spin_ineff ~ is_post6 + velo_z"
    model = smf.mixedlm(formula, data=dfm, groups=dfm["pitcher"])
    fit = model.fit(reml=False, method="lbfgs", maxiter=250, disp=False)

    coef = float(fit.params.get("is_post6", np.nan))
    pval = float(fit.pvalues.get("is_post6", np.nan))
    ci = fit.conf_int().loc["is_post6"]
    ci_low, ci_high = float(ci[0]), float(ci[1])

    return formula, coef, pval, (ci_low, ci_high)

def make_research_brief_pdf(
    n_pitchers: int,
    t_stat: float,
    p_val: float,
    cohens_d: float,
    r: float,
    rp: float,
    formula: str,
    coef_post6: float,
    p_post6: float,
    ci_low: float,
    ci_high: float,
    fig_path: str
):
    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=44
    )

    title_style = ParagraphStyle(name="T", fontSize=18, leading=22, spaceAfter=4)
    sub_style = ParagraphStyle(name="S", fontSize=11, leading=14, textColor=colors.HexColor("#444444"), spaceAfter=8)
    h_style = ParagraphStyle(name="H", fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=4)
    b_style = ParagraphStyle(name="B", fontSize=10.8, leading=13.5)
    stat_style = ParagraphStyle(name="STAT", fontSize=10.8, leading=13.5, textColor=colors.HexColor("#0B3D91"))

    elements = []
    elements.append(Paragraph(f"<b>Spin Inefficiency in Day vs Night MLB Games — {YEAR}</b>", title_style))
    elements.append(Paragraph("June–July | 4-Seam Fastballs | Wind ≤ 5 mph | ET Buckets (Pre-2pm vs Post-6pm)", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#C8C8C8")))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Methods</b>", h_style))
    elements.append(Paragraph(
        f"Within-pitcher comparison on <b>{n_pitchers}</b> MLB pitchers (≥{MIN_PITCHES_PER_BUCKET} pitches per bucket). "
        "Spin efficiency is inferred via Magnus-based <b>Option A</b> (movement + velocity; not directly measured). "
        "Primary model: pitch-level mixed-effects regression with random intercept per pitcher, controlling for standardized velocity.",
        b_style
    ))

    elements.append(Paragraph("<b>Results</b>", h_style))

    table_data = [
        ["Paired t-test (pitcher means)", f"t = {t_stat:.3f}", f"p = {p_val:.3f}", f"Cohen’s d = {cohens_d:.3f}"],
        ["Correlation (Δvelo vs Δspin ineff)", f"r = {r:.3f}", f"p = {rp:.3f}", ""],
        ["Mixed-effects (pitch-level)", f"β(post-6pm) = {coef_post6:.5f}", f"p = {p_post6:.4f}",
         f"95% CI [{ci_low:.5f}, {ci_high:.5f}]"],
    ]
    tbl = Table(table_data, colWidths=[2.55*inch, 2.05*inch, 1.15*inch, 1.95*inch])
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 10.2),
        ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#E0E0E0")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#F7F7F7"), colors.white]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"<b>Model:</b> {formula}  +  (1 | pitcher)", stat_style))

    elements.append(Paragraph("<b>Interpretation</b>", h_style))
    elements.append(Paragraph(
        "Pitcher-mean differences were not statistically significant. "
        "The pitch-level mixed model detects a <b>very small</b> post-6pm reduction in inferred spin inefficiency after controlling for velocity "
        "(~0.7 percentage points). Given Option A is an inferred metric and ET bucketing is not stadium-local, "
        "treat this as a modest association—not a definitive physiological claim.",
        b_style
    ))

    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#C8C8C8")))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Figure — Spin Inefficiency Δ by Pitcher (Sorted)</b>", h_style))
    elements.append(RLImage(fig_path, width=6.85*inch, height=3.25*inch))

    doc.build(elements)
    print("\nPDF Generated:", OUT_PDF)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Cache file not found: {CACHE_FILE}\n"
            "Run your June–July cache-building script once, then rerun this."
        )

    df = pd.read_parquet(CACHE_FILE)
    print("Loaded rows:", len(df))

    df_sel = select_pitchers_strict(df)
    pivot = build_pitcher_summary(df_sel)
    pivot.to_csv(OUT_CSV, index=False)

    # Paired t-test on pitcher means
    t_stat, p_val = ttest_rel(pivot["spin_ineff_avg_post_6pm"], pivot["spin_ineff_avg_pre_2pm"])
    diff = pivot["spin_ineff_avg_post_6pm"] - pivot["spin_ineff_avg_pre_2pm"]
    sd = diff.std(ddof=1)
    cohens_d = float(diff.mean() / sd) if (sd is not None and sd != 0 and not np.isnan(sd)) else float("nan")

    # Correlation
    r, rp = pearsonr(pivot["velo_diff"].values, pivot["spin_ineff_diff"].values)

    # MixedLM
    formula, coef_post6, p_post6, (ci_low, ci_high) = mixedlm_pitch_level(df_sel)

    # Figure + PDF
    fig_path = make_sorted_bar(pivot)
    make_research_brief_pdf(
        n_pitchers=len(pivot),
        t_stat=float(t_stat),
        p_val=float(p_val),
        cohens_d=float(cohens_d),
        r=float(r),
        rp=float(rp),
        formula=formula,
        coef_post6=float(coef_post6),
        p_post6=float(p_post6),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        fig_path=fig_path
    )

    print("\nSaved CSV:", OUT_CSV)
    print("Saved figure:", OUT_FIG)