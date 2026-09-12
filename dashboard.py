"""
football_props dashboard.

    streamlit run dashboard.py

Four views, each answering one question in plain language:

    This Week     who does the model like, and by how much?
    vs The Book   where do we disagree with the price, and is that real?
    Track Record  when the games were played, was it right?
    Model Health  what is known to be wrong right now?

DESIGN NOTES
------------
**The honesty strip is not decoration.** The model has no demonstrated
ability to beat a closing line -- one captured week, 596 rows, as of
2026-09-12. A dashboard that opens with "TOP PICKS" in green would be
lying about that, so the state of the evidence is the first thing on the
page and travels with every edge shown. Projections are presented as
projections; disagreements are presented as disagreements.

**Charts appear where the question is about shape, tables where it is
about lookup.** "How much more does the model like Nacua than Kupp" is a
magnitude-by-identity question and gets a horizontal bar. "What does it
say about this one guy" is lookup and gets a table. Calibration is a line
against the identity diagonal because that is what calibration IS.

**Palette** -- slots 1/2/3 of the validated categorical order (blue,
orange, aqua) for model / market / climatology, plus the fixed status
palette for state. Validated with the dataviz validator in light mode:
worst adjacent CVD dE 9.2, normal-vision dE 27.6, all gates pass. Aqua
carries a contrast WARN against the light surface, so every aqua mark
ships a visible label -- which the relief rule requires and which these
charts do anyway.
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

st.set_page_config(page_title="Football Props", page_icon="🏈", layout="wide")

# --- palette -----------------------------------------------------------
C_MODEL, C_MARKET, C_CLIM = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e8e7e3"

SLATE_DIR, CACHE_DIR = "slates", "cache"

PROP_LABEL = {
    "receiving_yards": "Receiving Yards",
    "receptions": "Receptions",
    "rushing_yards": "Rushing Yards",
    "passing_yards": "Passing Yards",
    "receiving_tds": "Receiving TDs",
    "rushing_tds": "Rushing TDs",
}
PROP_UNIT = {
    "receiving_yards": "yds", "rushing_yards": "yds", "passing_yards": "yds",
    "receptions": "rec", "receiving_tds": "TD", "rushing_tds": "TD",
}

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.015em; }}

  .hdr {{ display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
         margin-bottom:.15rem; }}
  .hdr h1 {{ margin:0; font-size:1.9rem; }}
  .hdr .wk {{ font-size:.95rem; color:{INK_2}; }}

  .strip {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.9rem 0 1.3rem; }}
  .pill {{ display:inline-flex; align-items:center; gap:.4rem;
          padding:.32rem .7rem; border-radius:999px; font-size:.8rem;
          font-weight:600; border:1px solid transparent; }}
  .pill.ok   {{ background:#eaf7ea; color:#0a6b0a; border-color:#bfe4bf; }}
  .pill.warn {{ background:#fdf3dc; color:#7a5600; border-color:#f0d99a; }}
  .pill.crit {{ background:#fbe9e9; color:#8f2020; border-color:#f0bfbf; }}
  .pill.info {{ background:#eaf1fb; color:#1b4c8f; border-color:#c3d8f4; }}

  .cards {{ display:grid; gap:.75rem;
           grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
           margin-bottom:1.1rem; }}
  .card {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:12px;
          padding:.8rem .95rem; }}
  .card .k {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
             color:{INK_3}; font-weight:700; }}
  .card .v {{ font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;
             font-variant-numeric:tabular-nums; }}
  .card .s {{ font-size:.76rem; color:{INK_2}; }}

  .lede {{ background:#f7f6f3; border-left:3px solid {C_MODEL};
          padding:.7rem .95rem; border-radius:0 8px 8px 0; font-size:.9rem;
          color:{INK_2}; margin-bottom:1rem; }}
  .lede b {{ color:{INK}; }}

  .rowlab {{ font-variant-numeric:tabular-nums; }}
  code {{ font-size:.85em; }}
  [data-testid="stMetricValue"] {{ font-size:1.4rem; }}
</style>
"""


# --- loading -----------------------------------------------------------

@st.cache_data(ttl=300)
def list_slates() -> list:
    return sorted(glob.glob(os.path.join(SLATE_DIR, "slate_*.parquet")), reverse=True)


@st.cache_data(ttl=300)
def load_parquet(path: str):
    if not os.path.exists(path):
        return None
    return pl.read_parquet(path).to_pandas()


def season_week(path: str):
    m = re.search(r"slate_(\d+)_wk(\d+)\.parquet$", path)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def fmt(prop: str) -> str:
    return PROP_LABEL.get(prop, prop.replace("_", " ").title())


def pill(kind: str, text: str) -> str:
    return f'<span class="pill {kind}">{text}</span>'


def cards(items) -> None:
    """items: (label, value, sub) triples."""
    html = '<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def brier_skill(y, p) -> float:
    b = float(np.mean(y))
    d = b * (1 - b)
    return float("nan") if d <= 0 else 1 - float(np.mean((p - y) ** 2)) / d


# --- charts ------------------------------------------------------------

def leaderboard_chart(df: pd.DataFrame, value: str, label: str,
                      unit: str, color: str = C_MODEL, height: int = None):
    """
    Horizontal bars: magnitude by identity, which is what a leaderboard is.
    Every bar is directly labelled, so identity and value never depend on
    reading a colour or chasing an axis.
    """
    df = df.copy()
    df["_lab"] = df[value].map(lambda v: f"{v:,.1f} {unit}" if unit != "TD"
                               else f"{v:.0%}")
    h = height or max(200, 26 * len(df) + 30)
    # Headroom for the direct labels. Without it the longest bar's label is
    # clipped at the plot edge -- and the direct label is the whole reason
    # these bars are readable without chasing the axis.
    top = float(df[value].max())
    base = alt.Chart(df).encode(
        y=alt.Y("player_name:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=180, labelFontSize=12,
                              labelColor=INK, domain=False, ticks=False)),
        x=alt.X(f"{value}:Q", title=label,
                scale=alt.Scale(domain=[0, top * 1.18], nice=False),
                axis=alt.Axis(grid=True, gridColor=GRID, domainColor=GRID,
                              tickColor=GRID, labelColor=INK_3,
                              titleColor=INK_2)))
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                         color=color, height=16).encode(
        tooltip=[alt.Tooltip("player_name:N", title="Player"),
                 alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip(f"{value}:Q", title=label, format=",.2f")])
    text = base.mark_text(align="left", dx=6, fontSize=11, color=INK_2).encode(
        text="_lab:N")
    return (bars + text).properties(height=h).configure_view(stroke=None)


# --- views -------------------------------------------------------------

def view_week(path: str, df: pd.DataFrame):
    season, week = season_week(path)

    n_clean = int(df["clean"].sum()) if "clean" in df else 0
    flagged = int((df["availability"] != "CLEAR").sum()) if "availability" in df else 0
    stamp = df["predicted_at"].max() if "predicted_at" in df else None
    games = df["game_id"].nunique() if "game_id" in df else 0
    cards([
        ("Games", f"{games}", f"{df['team'].nunique()} teams"),
        ("Players", f"{df['player_id'].nunique():,}", f"{len(df):,} projections"),
        ("Locked in", f"{n_clean:,}", "committed before kickoff"),
        ("Injury-flagged", f"{flagged:,}", "rows carrying a report"),
        ("Built", str(stamp)[5:16] if stamp is not None else "—", "UTC"),
    ])

    if "availability" in df and flagged == 0:
        st.warning(
            "Every player reads CLEAR. If this week's injury report has not "
            "published yet that means the report is **absent**, not that "
            "everyone is healthy. Re-run after Wednesday.")

    props = [p for p in PROP_LABEL if p in set(df["prop"])]
    props += [p for p in sorted(set(df["prop"])) if p not in props]

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Who the model likes")
        c1, c2 = st.columns([3, 2])
        prop = c1.selectbox("Prop", props, format_func=fmt, key="lb_prop")
        topn = c2.slider("Show", 5, 30, 12, key="lb_n")

        sub = df[df["prop"] == prop].copy()
        if "availability" in sub.columns:
            healthy = st.checkbox("Hide injury-flagged players", value=False)
            if healthy:
                sub = sub[sub["availability"] == "CLEAR"]
        top = sub.nlargest(topn, "expected")
        if top.empty:
            st.info("Nothing projected for this prop.")
        else:
            st.altair_chart(
                leaderboard_chart(top, "expected", fmt(prop),
                                  PROP_UNIT.get(prop, "")),
                use_container_width=True)
            st.caption(
                f"Projected {fmt(prop).lower()} — the model's **average** "
                f"outcome, not its most likely one. These distributions are "
                f"right-skewed, so the median sits below the mean.")

    with right:
        st.subheader("Most likely to score")
        td = load_parquet(path.replace("slate_", "anytime_td_"))
        if td is None or td.empty:
            st.info("No anytime-TD file for this slate.")
        else:
            t = td.nlargest(12, "anytime_td").copy()
            st.altair_chart(
                leaderboard_chart(t, "anytime_td", "P(scores a TD)", "TD",
                                  color=C_CLIM),
                use_container_width=True)
            st.caption(
                "P(anytime) = 1 − P(no receiving TD) × P(no rushing TD). The "
                "two are treated as independent, which slightly **understates** "
                "goal-line backs — the players the book prices most sharply.")

    st.divider()
    st.subheader("Look up a player")
    f1, f2, f3 = st.columns([2, 2, 3])
    fprop = f1.multiselect("Prop", props, format_func=fmt, key="tbl_prop")
    fteam = f2.multiselect("Team", sorted(df["team"].unique()), key="tbl_team")
    search = f3.text_input("Name contains", key="tbl_q",
                           placeholder="e.g. Nacua")

    t = df
    if fprop:
        t = t[t["prop"].isin(fprop)]
    if fteam:
        t = t[t["team"].isin(fteam)]
    if search:
        t = t[t["player_name"].str.contains(search, case=False, na=False)]

    show = t.copy()
    show["prop"] = show["prop"].map(fmt)
    cols = [c for c in ["player_name", "team", "position", "prop", "expected",
                        "exp_opportunities", "availability"] if c in show.columns]
    line_cols = [c for c in show.columns if c.startswith("over_")]
    show = show.sort_values("expected", ascending=False)[cols + line_cols]
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=420,
        column_config={
            "player_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "position": st.column_config.TextColumn("Pos", width="small"),
            "prop": st.column_config.TextColumn("Prop", width="medium"),
            "expected": st.column_config.NumberColumn("Projected", format="%.1f"),
            "exp_opportunities": st.column_config.NumberColumn(
                "Opps", format="%.1f",
                help="Targets, carries or attempts the model expects"),
            "availability": st.column_config.TextColumn("Status", width="small"),
            **{c: st.column_config.ProgressColumn(
                f"o{c[5:]}", min_value=0.0, max_value=1.0, format="%.2f")
               for c in line_cols},
        })
    st.caption(
        "`o24.5` is the model's probability of going **over** that line. "
        "Columns are empty for props where the line does not apply.")


def view_market(path: str):
    season, week = season_week(path)
    edges = load_parquet(os.path.join(CACHE_DIR, f"edges_{season}_wk{week}.parquet"))
    if edges is None or edges.empty:
        st.info(
            "No captured lines for this week yet.\n\n"
            "Run **`python run_slate.py`** on Wednesday — it buys the lines, "
            "builds the slate and computes this page in one go. Props post "
            "midweek; before that the markets come back empty, which is not "
            "an error.")
        return

    mean_edge = float(edges["edge"].mean())
    st.markdown(
        '<div class="lede">'
        "<b>Read this page as a bug-finder, not a bet list.</b> A large average "
        "edge almost always means a units or scope problem rather than free "
        "money — and the current average is far from zero. Two known causes are "
        "already measured: the model's distributions are right-skewed by ~4.5 "
        "units, so a line near the expectation correctly prices below 0.50; and "
        "the book's lines sit ~2.4 above the model's mean, which is a real "
        "level gap in the book's favour."
        "</div>", unsafe_allow_html=True)

    cards([
        ("Priced props", f"{len(edges):,}",
         f"{edges['player_id'].nunique()} players"),
        ("Average edge", f"{mean_edge:+.3f}",
         "model probability − book probability"),
        ("Book hold", f"{float(edges['hold'].mean()):.2%}",
         "the vig, already removed"),
        ("Books quoted", f"{float(edges['n_books'].mean()):.1f}",
         "average per line"),
    ])

    dist = edges[["edge"]].copy()
    hist = alt.Chart(dist).mark_bar(color=C_MODEL, opacity=0.85).encode(
        x=alt.X("edge:Q", bin=alt.Bin(maxbins=40), title="Edge (model − book)",
                axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                              labelColor=INK_3, titleColor=INK_2, format="+.2f")),
        y=alt.Y("count():Q", title="Priced props",
                axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                              labelColor=INK_3, titleColor=INK_2)),
        tooltip=[alt.Tooltip("count():Q", title="props")])
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=INK_3, strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    st.altair_chart((hist + zero).properties(height=250)
                    .configure_view(stroke=None), use_container_width=True)
    st.caption(
        "Dashed line is perfect agreement. A healthy distribution is centred "
        "there and two-sided; this one is shifted left, which is the open "
        "question week 1's results will settle.")

    st.subheader("Biggest disagreements")
    side = st.radio("Direction", ["Model says OVER", "Model says UNDER"],
                    horizontal=True, label_visibility="collapsed")
    e = edges.copy()
    e["prop"] = e["prop"].map(fmt)
    e = e.nlargest(15, "edge") if side.endswith("OVER") else e.nsmallest(15, "edge")
    e = e.sort_values("edge", ascending=not side.endswith("OVER"))

    st.dataframe(
        e[["matched_name", "team", "prop", "line", "model_prob", "market_prob",
           "edge", "n_books", "availability"]],
        use_container_width=True, hide_index=True,
        column_config={
            "matched_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "prop": st.column_config.TextColumn("Prop", width="medium"),
            "line": st.column_config.NumberColumn("Line", format="%.1f"),
            "model_prob": st.column_config.ProgressColumn(
                "Model", min_value=0.0, max_value=1.0, format="%.3f"),
            "market_prob": st.column_config.ProgressColumn(
                "Book", min_value=0.0, max_value=1.0, format="%.3f"),
            "edge": st.column_config.NumberColumn("Edge", format="%+.3f"),
            "n_books": st.column_config.NumberColumn("Books", format="%d"),
            "availability": st.column_config.TextColumn("Status", width="small"),
        })
    st.caption(
        "A single-book line is a much weaker number than a five-book "
        "consensus. Check the **Books** column before taking any row "
        "seriously.")


def view_record():
    log = load_parquet(os.path.join(CACHE_DIR, "scoring_log.parquet"))
    if log is None or log.empty:
        st.info("Nothing scored yet. Run **`python score_slate.py`** after a "
                "week's games are played.")
        return

    weeks = (log[["season", "week"]].drop_duplicates()
                .sort_values(["season", "week"]))
    wk_label = ", ".join(f"{int(r.season)} wk{int(r.week)}"
                         for r in weeks.itertuples())
    cards([
        ("Weeks scored", f"{len(weeks)}", wk_label or "—"),
        ("Rows", f"{len(log):,}", "player-props with a result"),
        ("Players", f"{log['player_id'].nunique():,}", "distinct"),
        ("Props covered", f"{log['prop'].nunique()}", "of 6"),
    ])

    st.warning(
        "**These rows predate the 2026-09-12 model changes.** The share "
        "dispersion fix and the QB mixture both landed after this week was "
        "scored, so the figures below describe an older model and are not "
        "comparable with weeks scored from now on. `model_flags.py` records "
        "the changeover.", icon="⚠️")

    # PER PROP, NEVER POOLED. A mean bias across receiving yards and
    # receptions averages yards with catches and means nothing -- the first
    # version of this page printed exactly that number and it was -1.59.
    per = (log.assign(err=log["expected"] - log["actual"])
              .groupby("prop")
              .agg(rows=("actual", "size"), actual=("actual", "mean"),
                   projected=("expected", "mean"), bias=("err", "mean"),
                   crps=("crps", "mean"))
              .reset_index())
    per["ratio"] = per["projected"] / per["actual"].replace(0, np.nan)
    per["prop"] = per["prop"].map(fmt)
    st.dataframe(
        per[["prop", "rows", "actual", "projected", "ratio", "bias", "crps"]],
        use_container_width=True, hide_index=True,
        column_config={
            "prop": st.column_config.TextColumn("Prop", width="medium"),
            "rows": st.column_config.NumberColumn("Rows", format="%d"),
            "actual": st.column_config.NumberColumn("Actual avg", format="%.2f"),
            "projected": st.column_config.NumberColumn("Projected avg",
                                                       format="%.2f"),
            "ratio": st.column_config.NumberColumn(
                "Ratio", format="%.3f",
                help="Projected ÷ actual. 1.00 is perfect on the mean."),
            "bias": st.column_config.NumberColumn("Bias", format="%+.2f"),
            "crps": st.column_config.NumberColumn(
                "CRPS", format="%.3f",
                help="Distribution error. Comparable between versions of the "
                     "model, not between props."),
        })
    st.caption(
        "Every figure is **per prop**. Averaging bias or CRPS across props "
        "would mix yards with catches, and the number would mean nothing.")

    props = sorted(log["prop"].unique())
    prop = st.selectbox("Prop", props, format_func=fmt, key="rec_prop")
    sub = log[log["prop"] == prop].copy()

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("**Projected vs actual**")
        pts = sub.dropna(subset=["expected", "actual"])
        if len(pts) > 3000:
            pts = pts.sample(3000, random_state=0)
        if pts.empty:
            st.info("No rows.")
        else:
            lim = float(max(pts["expected"].max(), pts["actual"].max())) * 1.05
            sc = alt.Chart(pts).mark_circle(size=34, opacity=0.42,
                                            color=C_MODEL).encode(
                x=alt.X("expected:Q", title="projected",
                        scale=alt.Scale(domain=[0, lim]),
                        axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                                      labelColor=INK_3, titleColor=INK_2)),
                y=alt.Y("actual:Q", title="actual",
                        scale=alt.Scale(domain=[0, lim]),
                        axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                                      labelColor=INK_3, titleColor=INK_2)),
                tooltip=["player_name", "team",
                         alt.Tooltip("expected:Q", format=".1f"),
                         alt.Tooltip("actual:Q", format=".1f")])
            diag = alt.Chart(pd.DataFrame({"x": [0, lim]})).mark_line(
                strokeDash=[5, 4], color=INK_3, strokeWidth=1).encode(
                x="x:Q", y="x:Q")
            st.altair_chart((sc + diag).properties(height=330)
                            .configure_view(stroke=None),
                            use_container_width=True)
            st.caption("Points on the dashed line were predicted exactly. "
                       "A cloud tilted below it means over-prediction.")

    with b:
        st.markdown("**Calibration at a line**")
        lcols = [c for c in sub.columns
                 if c.startswith("over_") and sub[c].notna().any()]
        if not lcols:
            st.info("No probability columns for this prop.")
        else:
            lc = st.selectbox("Line", lcols, format_func=lambda c: f"over {c[5:]}",
                              key="rec_line")
            L = float(lc[5:])
            s2 = sub.dropna(subset=[lc])
            y = (s2["actual"] > L).astype(float).values
            p = s2[lc].values
            if len(y) < 30 or y.mean() in (0.0, 1.0):
                st.info("Too few rows at this line to say anything.")
            else:
                rows = []
                nb = 6 if len(p) >= 300 else 4
                q = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
                for lo, hi in zip(q[:-1], q[1:]):
                    m = (p >= lo) & (p <= hi if hi == q[-1] else p < hi)
                    if m.sum() < 10:
                        continue
                    rows.append({"predicted": float(p[m].mean()),
                                 "observed": float(y[m].mean()),
                                 "n": int(m.sum())})
                if len(rows) < 2:
                    st.info(
                        f"Only {len(s2)} scored rows at this line -- not "
                        f"enough distinct probabilities to bin yet. "
                        f"Calibration needs a few hundred.")
                else:
                    cal = pd.DataFrame(rows)
                    base = alt.Chart(cal).encode(
                        x=alt.X("predicted:Q", title="model said",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                                              labelColor=INK_3, titleColor=INK_2,
                                              format=".0%")),
                        y=alt.Y("observed:Q", title="actually happened",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                                              labelColor=INK_3, titleColor=INK_2,
                                              format=".0%")))
                    diag = alt.Chart(pd.DataFrame({"x": [0, 1]})).mark_line(
                        strokeDash=[5, 4], color=INK_3, strokeWidth=1).encode(
                        x="x:Q", y="x:Q")
                    ln = base.mark_line(color=C_MODEL, strokeWidth=2)
                    pt = base.mark_point(size=95, filled=True, color=C_MODEL,
                                         stroke=SURFACE, strokeWidth=2).encode(
                        tooltip=[alt.Tooltip("predicted:Q", format=".1%"),
                                 alt.Tooltip("observed:Q", format=".1%"), "n"])
                    st.altair_chart((diag + ln + pt).properties(height=330)
                                    .configure_view(stroke=None),
                                    use_container_width=True)
                    st.metric("Brier skill vs the base rate",
                              f"{brier_skill(y, p):+.4f}",
                              help="1 − brier/(p(1−p)). Zero means no better "
                                   "than always guessing the base rate. Pooled, "
                                   "never averaged per week.")


def view_health():
    st.subheader("What is known to be wrong right now")
    st.markdown(
        '<div class="lede">'
        "Every number here was measured, not estimated. <b>CONTEXT.md</b> "
        "carries the working for each one, including the versions that did not "
        "survive contact with the data."
        "</div>", unsafe_allow_html=True)

    issues = pd.DataFrame([
        {"Area": "Mean, players who get lines", "State": "good",
         "Reading": "ratio 0.99–1.01",
         "What it means": "Projections are right on average for the players a "
                          "book actually prices."},
        {"Area": "Distribution width, receivers", "State": "good",
         "Reading": "variance ratio 0.87",
         "What it means": "Slightly over-confident. Under-prices the far tail "
                          "a little."},
        {"Area": "Distribution width, QBs", "State": "good",
         "Reading": "QB1 1.03, QB2 0.80",
         "What it means": "Fixed by the empirical share mixture on 2026-09-12. "
                          "Was 0.71 and 0.46."},
        {"Area": "Deep depth-chart ranks", "State": "warning",
         "Reading": "WR7 2.5×, RB4 2.5×",
         "What it means": "Over-predicts reserves. Books do not price these "
                          "players, so it costs correctness, not money."},
        {"Area": "Beating the book", "State": "critical",
         "Reading": "unmeasured",
         "What it means": "One captured week. Needs ~12–15 scored weeks before "
                          "the question can even be asked."},
        {"Area": "Availability flag", "State": "warning",
         "Reading": "logged, feeds nothing",
         "What it means": "Deliberately inert until ~20 scored weeks say "
                          "whether it earns a place."},
    ])
    # Rendered as markdown rather than a dataframe so the explanation column
    # WRAPS. In a dataframe it truncated mid-sentence, which turns the one
    # column that carries the meaning into decoration.
    ICON = {"good": "🟢", "warning": "🟡", "critical": "🔴"}
    md = ["| | Area | Reading | What it means |", "|---|---|---|---|"]
    # by column NAME, not itertuples -- "What it means" has a space, so
    # itertuples renames it to a positional _4 that silently shifts if a
    # column is ever inserted before it.
    for _, r in issues.iterrows():
        md.append(f"| {ICON[r['State']]} | **{r['Area']}** | `{r['Reading']}` | "
                  f"{r['What it means']} |")
    st.markdown("\n".join(md))
    st.caption(
        "The icon and the wording both carry the state, so it never depends "
        "on colour alone.")

    st.divider()
    st.subheader("The weekly routine")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Wednesday**")
        st.code("python run_slate.py", language="powershell")
        st.caption(
            "Buys this week's lines (skipping any it already has), builds the "
            "slate, computes the disagreements. Safe to re-run — it spends "
            "nothing twice.")
    with c2:
        st.markdown("**Tuesday, after the Monday game**")
        st.code("python score_slate.py", language="powershell")
        st.caption(
            "Finds the newest week that is actually finished and scores it. "
            "Refuses a half-played week rather than guessing.")

    slates = list_slates()
    if slates:
        df = load_parquet(slates[0])
        if df is not None and "availability" in df:
            st.divider()
            st.subheader("Injury report, current slate")
            counts = (df.groupby("availability")["player_id"].nunique()
                        .reset_index(name="players")
                        .sort_values("players", ascending=False))
            ch = alt.Chart(counts).mark_bar(
                color=C_MARKET, cornerRadiusTopRight=4,
                cornerRadiusBottomRight=4, height=18).encode(
                y=alt.Y("availability:N", sort="-x", title=None,
                        axis=alt.Axis(labelColor=INK, domain=False, ticks=False)),
                x=alt.X("players:Q", title="players",
                        scale=alt.Scale(
                            domain=[0, float(counts["players"].max()) * 1.15],
                            nice=False),
                        axis=alt.Axis(gridColor=GRID, domainColor=GRID,
                                      labelColor=INK_3, titleColor=INK_2)),
                tooltip=["availability", "players"])
            lab = alt.Chart(counts).mark_text(
                align="left", dx=6, fontSize=11, color=INK_2).encode(
                y=alt.Y("availability:N", sort="-x"), x="players:Q",
                text="players:Q")
            st.altair_chart((ch + lab).properties(height=max(120, 30 * len(counts)))
                            .configure_view(stroke=None),
                            use_container_width=True)


# --- app ---------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    slates = list_slates()
    if not slates:
        st.title("🏈 Football Props")
        st.info("No slate yet. Run **`python run_slate.py`** to build one.")
        return

    with st.sidebar:
        st.markdown("### Slate")
        path = st.selectbox(
            "Week", slates, label_visibility="collapsed",
            format_func=lambda p: (lambda sw: f"{sw[0]} · Week {sw[1]}")(
                season_week(p)))
        st.caption("Older weeks stay on disk exactly as they were committed.")

    df = load_parquet(path)
    season, week = season_week(path)

    st.markdown(
        f'<div class="hdr"><h1>🏈 Football Props</h1>'
        f'<span class="wk">{season} · Week {week}</span></div>',
        unsafe_allow_html=True)

    edges_path = os.path.join(CACHE_DIR, f"edges_{season}_wk{week}.parquet")
    log = load_parquet(os.path.join(CACHE_DIR, "scoring_log.parquet"))
    n_weeks = 0 if log is None or log.empty else len(
        log[["season", "week"]].drop_duplicates())

    st.markdown(
        '<div class="strip">'
        + pill("crit", "◆ No proven edge over the book")
        + pill("info", f"◆ {n_weeks} week{'s' if n_weeks != 1 else ''} scored "
                       f"of ~12–15 needed")
        + pill("ok" if os.path.exists(edges_path) else "warn",
               ("◆ Lines captured" if os.path.exists(edges_path)
                else "◆ No lines captured this week"))
        + pill("ok", "◆ Projections calibrated on the mean")
        + "</div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["This Week", "vs The Book", "Track Record", "Model Health"])
    with t1:
        if df is None or df.empty:
            st.warning("Empty slate.")
        else:
            view_week(path, df)
    with t2:
        view_market(path)
    with t3:
        view_record()
    with t4:
        view_health()


if __name__ == "__main__":
    main()
