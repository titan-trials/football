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

def games_panel(df: pd.DataFrame, season: int, week: int):
    """
    The week's actual schedule, with what has already happened.

    THIS PANEL EXISTS BECAUSE THE PAGE WAS CONFUSING WITHOUT IT. It said
    "16 games" and stopped, which invites the reasonable question "but
    aren't some of those today?" An NFL week is not a calendar week -- week
    1 of 2026 runs Wednesday the 9th through Monday the 14th -- so a slate
    routinely holds games that are finished, games kicking off in an hour,
    and games two days away, all at once.
    """
    if "kickoff" not in df.columns or "game_id" not in df.columns:
        return
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    g = (df.dropna(subset=["kickoff"])
           .groupby("game_id")
           .agg(kickoff=("kickoff", "min"),
                teams=("team", lambda s: " @ ".join(sorted(set(s))[:2])),
                players=("player_id", "nunique"))
           .reset_index()
           .sort_values("kickoff"))
    if g.empty:
        return

    # Kicked off more than ~3.5 hours ago is over; inside that window it is
    # being played right now. An NFL game runs about three hours.
    mins = (now - g["kickoff"]).dt.total_seconds() / 60.0
    g["status"] = np.where(mins > 210, "Final",
                  np.where(mins >= 0, "Playing now", "Upcoming"))
    g["when"] = g["kickoff"].dt.strftime("%a %d %b · %H:%M UTC")
    g["in_hours"] = (-mins / 60.0).round(1)

    n_up = int((g["status"] == "Upcoming").sum())
    n_now = int((g["status"] == "Playing now").sum())
    n_done = int((g["status"] == "Final").sum())

    # <b>, not ** -- this string goes inside an HTML block, where markdown
    # bold renders as literal asterisks.
    bits = []
    if n_done:
        bits.append(f"<b>{n_done} already played</b>")
    if n_now:
        bits.append(f"<b>{n_now} being played right now</b>")
    if n_up:
        nxt = g[g["status"] == "Upcoming"].iloc[0]
        bits.append(f"<b>{n_up} still to come</b> — next in "
                    f"{nxt['in_hours']:.0f}h ({nxt['when']})")
    st.markdown(
        '<div class="lede">'
        f"An NFL week is not a calendar week. <b>Week {week}</b> runs "
        f"{g['kickoff'].min():%a %d %b} to {g['kickoff'].max():%a %d %b}, and "
        f"right now: " + ", ".join(bits) + ". Every one of these games is in "
        "the slate below."
        "</div>", unsafe_allow_html=True)

    ICON = {"Final": "✓", "Playing now": "●", "Upcoming": "○"}
    g["st"] = g["status"].map(ICON) + "  " + g["status"]
    st.dataframe(
        g[["st", "teams", "when", "players"]],
        use_container_width=True, hide_index=True, height=min(430, 38 * len(g) + 40),
        column_config={
            "st": st.column_config.TextColumn("Status", width="small"),
            "teams": st.column_config.TextColumn("Game", width="small"),
            "when": st.column_config.TextColumn("Kickoff", width="medium"),
            "players": st.column_config.NumberColumn(
                "Players projected", format="%d"),
        })
    st.caption(
        "Times are UTC — Eastern is UTC−4 in September, Pacific UTC−7. "
        "Projections for a game that has already kicked off are frozen as "
        "they were committed beforehand; re-running never overwrites them.")


def view_week(path: str, df: pd.DataFrame):
    season, week = season_week(path)

    n_clean = int(df["clean"].sum()) if "clean" in df else 0
    flagged = int((df["availability"] != "CLEAR").sum()) if "availability" in df else 0
    stamp = df["predicted_at"].max() if "predicted_at" in df else None
    games = df["game_id"].nunique() if "game_id" in df else 0
    cards([
        ("Games", f"{games}", f"{df['team'].nunique()} teams this week"),
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

    st.subheader("This week's games")
    games_panel(df, season, week)
    st.divider()

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
    if "kickoff" in show.columns:
        show["kicks"] = pd.to_datetime(show["kickoff"]).dt.strftime("%a %H:%M")
    cols = [c for c in ["player_name", "team", "kicks", "position", "prop",
                        "expected", "exp_opportunities", "availability"]
            if c in show.columns]
    line_cols = [c for c in show.columns if c.startswith("over_")]
    show = show.sort_values("expected", ascending=False)[cols + line_cols]
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=420,
        column_config={
            "player_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "kicks": st.column_config.TextColumn(
                "Kickoff", width="small", help="UTC"),
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


def _metric_primer():
    """
    The two ideas the whole scorecard rests on, each with a worked example.

    This block exists because the first version of this page asserted
    "variance ratio 0.87" and left the reader to work out what that meant.
    A number nobody can interpret is decoration.
    """
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            f"""
##### 1 · Level — *is the average right?*

```
ratio  =  average projected  ÷  average actual
```

**Worked example.** Across 3,759 top-3 receivers the model projected
**21.90** yards each. They actually averaged **21.97**.

```
21.90 ÷ 21.97  =  0.997
```

So the model is 0.3% low — about **0.07 yards per player**. A ratio of
1.00 is perfect, 1.20 means projecting 20% too high, 0.80 means 20% too
low.

*This is the easy half, and it is the half that was already fine.*
""")
    with b:
        st.markdown(
            f"""
##### 2 · Width — *is the uncertainty right?*

```
variance ratio  =  variance the model claimed
                   ÷  how far off it actually was²
```

**Worked example.** The model says a receiver's yards have a spread of
about **24** (variance 576). In reality its misses average about **25.8**
(variance 662).

```
576 ÷ 662  =  0.87        √0.87 = 0.93
```

So the spread is **7% too narrow**. Below 1.00 means over-confident:
the model thinks it knows more than it does.

*This is the half that was badly broken and is now mostly fixed.*
""")

    st.markdown(
        '<div class="lede">'
        "<b>Why width gets its own row when the average is already right.</b> "
        "A prop does not pay out on the average — it pays out on whether the "
        "player goes over a line. Those are different questions about the same "
        "distribution, and the second one is mostly about the spread."
        "</div>", unsafe_allow_html=True)

    st.markdown(
        """
Take a receiver the model projects at **60 yards**, with a line at **90**.

| | spread | how many spreads away 90 is | P(over 90) |
|---|---|---|---|
| model thinks | 24.0 | (90 − 60) ÷ 24.0 = 1.25 | **10.6%** |
| reality | 25.8 | (90 − 60) ÷ 25.8 = 1.16 | **12.2%** |

Same projection of 60 yards. Same line. But a 7% narrower spread prices
the over at 10.6% when it is worth 12.2% — and **every** over above the
projection is priced low the same way. The level check passes cleanly the
whole time, because the level is not what is wrong.
""")


ISSUES = [
    {
        "state": "good",
        "area": "Level — players who get lines",
        "reading": "ratio 0.99 – 1.01",
        "short": "Projections are right on average for the players a book "
                 "actually posts a line on.",
        "detail": """
**What is measured.** Average projected ÷ average actual, for players in
the top three at their position on the depth chart — roughly the set a
book prices. Measured on all of 2025, one week at a time, with the model
only ever seeing earlier weeks.

| prop | projected | actual | ratio |
|---|---|---|---|
| Receiving yards | 21.90 | 21.97 | 0.997 |
| Receptions | 1.99 | 2.01 | 0.992 |
| Rushing yards | 11.95 | 12.28 | 0.973 |
| Passing yards | 81.06 | 80.20 | 1.011 |

**Why it is sliced by depth-chart rank.** Two obvious alternatives are
both mistakes this project already made. Slicing by the *prediction* hides
the problem — if the model shrinks everything toward the middle, the bins
move with it and every bin looks fine. Slicing by the *outcome* ("players
who caught at least one pass") conditions on success and guarantees the
model looks low. Depth-chart rank is known before kickoff, so it is a fair
way to cut the data.

**What good looks like.** 0.97 – 1.03. It is there.
""",
    },
    {
        "state": "good",
        "area": "Width — receivers",
        "reading": "variance ratio 0.87",
        "short": "Slightly over-confident. Under-prices the far tail a little.",
        "detail": """
**What is measured.** The variance the model claims ÷ the variance of its
actual misses. 1.00 is honest; below 1.00 means the model's distributions
are too narrow.

**Where it came from.** This was **0.57** before 2026-09-12 — the spread
was 25% too small. The cause was structural: the model treated a player's
share of his team's targets as a fixed number, so his only source of
week-to-week variation was luck in how the ball happened to bounce. In
reality the share itself moves — game plan, matchup, who else is playing.

**The fix** was to let the share be a random quantity rather than a fixed
one (a beta-binomial instead of a binomial, if you want the name). That
took receivers from 0.57 to 0.87.

**Is 0.87 a problem?** Mildly. It means the far tail — big games, the
over on a high line — is priced a few percent low. It is on the list, but
it is now smaller than the things above it.
""",
    },
    {
        "state": "good",
        "area": "Width — quarterbacks",
        "reading": "QB1 1.03 · QB2 0.80",
        "short": "Fixed on 2026-09-12 by modelling a QB's share as "
                 "'probability of starting'. Was 0.71 and 0.46.",
        "detail": """
**Why QBs were the worst case.** A quarterback's share of his team's pass
attempts is not a normal-looking spread around an average. It is close to
all-or-nothing:

| | throws <10% | 10–80% | throws ≥80% |
|---|---|---|---|
| QB1 (starter) | 4.4% | 5.7% | **90.0%** |
| QB2 (backup) | **82.1%** | 8.4% | 9.5% |
| QB4 | 100% | — | — |

Read the QB2 row. His *average* share is 0.127, and he is at 0.127
**essentially never** — he throws nothing four weeks in five and the whole
game one week in ten. Any smooth bell-shaped distribution centred on his
average describes a week that does not happen. The mean comes out right
and every probability comes out wrong.

**The fix** was to stop assuming a shape and just use the real one: the
model now draws a QB's share from the actual historical distribution for
his depth-chart slot. His own level is preserved by mixing that
distribution with "throws nothing" or "takes every snap" as needed — which
is the honest reading anyway, since a QB's share is really a *probability
of starting*, not a fraction of a start.

**Result.** Passing-yards distribution error fell 14.6% on the players who
get lines, and the systematic under-pricing of QB overs disappeared:

| line | before | after | actually happened |
|---|---|---|---|
| over 174.5 | 19.8% | **25.6%** | 25.7% |
| over 224.5 | 11.2% | **16.6%** | 17.6% |

**QB4 is worth a look.** He has thrown a pass in zero of 137 recorded
games. The old model projected him for 8.59 passing yards.
""",
    },
    {
        "state": "warning",
        "area": "Deep depth-chart ranks",
        "reading": "WR7 2.5× · RB4 2.5×",
        "short": "Over-predicts deep reserves. Books do not price these "
                 "players, so it costs correctness rather than money.",
        "detail": """
**What is wrong.** For players buried on the depth chart, the model
projects roughly 2.5× what they actually get. A WR7 is projected for 0.090
targets and gets 0.036.

**Put that in proportion.** The error is **0.054 targets per player-week**
— about one extra target every twenty games. Books do not post lines on
seventh receivers, so this costs correctness, not money. It is a yellow
row rather than a red one for exactly that reason.

**What was tried and did not work** (2026-09-12). The obvious explanation
is stale history: a player at WR7 today was a WR3 a month ago and carries
a WR3 average. That is *real* — current WR7s held a mean rank of 3.56 over
their last eight games and produced at 2.33× the WR7 rate.

So the fix was built: express a player's usage as a multiple of whatever
role he held at the time, then apply that multiple to the role he holds
now. It worked mechanically — the raw rate fell from 0.455 to 0.144,
landing exactly on target — **and the prediction got worse**, 2.5× to
3.4×.

**Why.** The fix swaps "his old role's production" for "full credit for
the role he holds now", and a deep role is not worth full credit. Ranks
below about four turn over constantly, and a newly-arrived TE4 does not
get what a settled TE4 gets. Promotions moved up more than demotions moved
down.

**What that rules out.** The stale-history explanation is now closed by
measurement rather than argument. The remaining hypothesis is about the
depth chart itself: below about rank four it may simply not be
informative enough to carry a prediction at all. If so the answer is a
different input — snap share, say — not a better estimator.

*(The same change **helped** quarterbacks, where QB1 vs QB2 is a real
distinction rather than a slowly-updating line on a chart, so it ships
for QBs only.)*
""",
    },
    {
        "state": "critical",
        "area": "Beating the book",
        "reading": "unmeasured",
        "short": "One captured week. Needs ~12–15 scored weeks before the "
                 "question can even be asked.",
        "detail": """
**This is the only row that decides whether any of this is worth
anything, and it is blank.**

Everything else on this page compares the model to *itself* or to a weak
baseline. Beating a book is a different and much harder bar, and it has
not been attempted, because attempting it needs scored weeks and there is
one.

**What gets measured.** When a week finishes, the model's probability and
the book's de-vigged probability are compared **on identical rows** —
same players, same lines, both committed before kickoff so neither side
can be adjusted afterwards. The score is pooled Brier skill.

**Why ~12–15 weeks.** A week gives about 600 priced props. The gap worth
detecting is small — on the order of 0.01 to 0.02 in skill — while one
week of 600 rows carries an uncertainty of roughly ±0.04 around the
measured gap. Uncertainty shrinks with the square root of the sample, so
going from ±0.04 to ±0.012 takes roughly ten times the rows. That is
twelve to fifteen weeks, and there is no way to hurry it.

**What to expect.** The market probably wins. That is the normal result
and is not a failure — closing lines are very hard to beat. The question
is by how much, and whether the gap shrinks as the model improves.

**Until then:** treat this as a research instrument, not a source of bets.
""",
    },
    {
        "state": "warning",
        "area": "Availability flag",
        "reading": "logged, feeds nothing",
        "short": "Deliberately inert until ~20 scored weeks say whether it "
                 "earns a place.",
        "detail": """
**What it is.** Every player on every slate carries his injury-report
status — Questionable, Doubtful, and so on. The model **does not use it**.
It is written down and fed to nothing.

**Why on purpose.** A feature that sounds obviously useful is exactly the
kind that gets wired in on vibes and quietly makes things worse. Logging
it first costs nothing and buys a real answer later: after enough scored
weeks, check whether flagged players underperformed their projections. If
they did, wire it in. If not, delete the column. Nothing was ever at risk
either way.

**What the evidence says so far, and it is thin:**

| bucket | players | projected | actual | gap |
|---|---|---|---|---|
| no report listed | 776 | 0.774 | 0.705 | −0.069 |
| Questionable | 47 | 0.808 | 0.660 | **−0.149** |

Flagged players do miss badly — more than twice the gap. But they were
5.7% of the group, so removing them moved the overall number only from
−0.073 to −0.069. **Real per player, negligible in aggregate, n = 47.**
That is suggestive and it is not evidence.
""",
    },
]


def view_health():
    st.subheader("Is the model any good?")
    st.markdown(
        '<div class="lede">'
        "Short version: <b>the projections are trustworthy as projections</b> "
        "— right on average, and now roughly right about their own "
        "uncertainty. <b>Whether they beat a bookmaker is unknown</b>, because "
        "that takes scored weeks and there is one. Every number below was "
        "measured on real seasons, one week at a time, with the model only "
        "ever seeing earlier weeks."
        "</div>", unsafe_allow_html=True)

    st.markdown("### How to read the numbers")
    _metric_primer()

    st.divider()
    st.markdown("### The scorecard")
    st.caption("Click any row to open the full explanation.")

    ICON = {"good": "🟢", "warning": "🟡", "critical": "🔴"}
    WORD = {"good": "healthy", "warning": "known issue",
            "critical": "not yet measured"}
    for it in ISSUES:
        head = (f"{ICON[it['state']]}  **{it['area']}** — {it['reading']}  "
                f"· _{WORD[it['state']]}_")
        with st.expander(head, expanded=False):
            st.markdown(f"**{it['short']}**")
            st.markdown(it["detail"])
    st.caption(
        "The icon, the word and the wording all carry the state, so it never "
        "depends on colour alone.")

    st.divider()
    st.markdown("### What would change the verdict")
    st.markdown(
        "One number, and only one: **the model's pooled Brier skill against "
        "the book's, on identical priced rows.** It appears at the bottom of "
        "`python score_slate.py` once a week has been scored. Nothing else on "
        "this page — not the level, not the width, not climatology — settles "
        "whether the model is worth money."
    )

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
            st.markdown("### Injury report, current slate")
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
            st.caption(
                "Counted, shown, and fed to no model — see the scorecard row "
                "above for why.")


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
