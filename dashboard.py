"""
football_props dashboard.

    streamlit run dashboard.py

A GAME DAY picker at the top, then four tabs:

    Betting Board    the players with a posted line, for the chosen day
    All Projections  every player the model projects, for lookup
    Track Record     when the games were played, was it right?
    Model Health     what is known to be wrong right now?

THE DAY PICKER IS THE PRIMARY CONTROL. "I want to bet on Sunday, who do I
look at" is the question this exists to answer, and the previous version
could not be asked it: a week was one undifferentiated pile of 2,631 rows
with no way to narrow to a day.

BOARD AND PROJECTIONS ARE SEPARATE ON PURPOSE. The slate projects 632
players across six props; books price 157 of them across two. Mixing
those in one table meant scrolling past hundreds of unbettable rows to
reach the handful that matter.

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
#
# TWO SELECTED PALETTES, NOT ONE FLIPPED. The dark column is the same three
# hues re-stepped for a dark surface, each validated against that surface
# rather than assumed to carry over. Both pass every gate of the dataviz
# validator:
#
#   light (surface #fcfcfb)  worst adjacent CVD dE 9.2, normal-vision 27.6
#   dark  (surface #1a1a19)  worst adjacent CVD dE 9.4, normal-vision 26.5,
#                            and all three clear 3:1 contrast
#
# The light set was the only one that existed at first, hardcoded. On a
# dark Streamlit theme that painted near-black text and axis labels onto a
# near-black background -- invisible, which is what "it's all black and I
# can't see anything" meant.
PALETTES = {
    "light": dict(
        model="#2a78d6", market="#eb6834", clim="#1baf7a",
        surface="#fcfcfb", ink="#1a1a18", ink2="#52514e", ink3="#8a8a85",
        grid="#e8e7e3", panel="#f7f6f3",
    ),
    "dark": dict(
        model="#3987e5", market="#d95926", clim="#199e70",
        surface="#1a1a19", ink="#f3f2ee", ink2="#c3c2b7", ink3="#93928a",
        grid="#383835", panel="#232321",
    ),
}
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

# Position colours, so a table of names is not one undifferentiated block.
# Always paired with the position's letters -- hue never carries it alone.
POS_DOT = {"QB": "🔵", "RB": "🟠", "WR": "🟢", "TE": "🟣", "FB": "🟡"}


def position_tag(pos: str) -> str:
    return f"{POS_DOT.get(pos, '⚪')} {pos}" if isinstance(pos, str) else ""

# ANCHORED TO THIS FILE, NOT TO THE WORKING DIRECTORY.
#
# `"slates"` and `"cache"` are relative paths, so they resolve against
# wherever streamlit happened to be launched from. Launch it from anywhere
# but the project root -- a shortcut, a parent folder, an IDE run button --
# and the page loads, finds nothing, and reports "no posted lines, run
# run_slate.py" while the files sit right there. A read-only dashboard
# should not care what directory you were standing in.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLATE_DIR = os.path.join(BASE_DIR, "slates")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

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

# Set by apply_theme() before anything renders. Every chart reads these at
# CALL time, so swapping the palette swaps the whole page.
C_MODEL = C_MARKET = C_CLIM = SURFACE = INK = INK_2 = INK_3 = GRID = PANEL = ""

PILLS = {
    "light": dict(ok_bg="#eaf7ea", ok_fg="#0a6b0a", ok_bd="#bfe4bf",
                  warn_bg="#fdf3dc", warn_fg="#7a5600", warn_bd="#f0d99a",
                  crit_bg="#fbe9e9", crit_fg="#8f2020", crit_bd="#f0bfbf",
                  info_bg="#eaf1fb", info_fg="#1b4c8f", info_bd="#c3d8f4"),
    # Dark: tinted backgrounds near the surface, text light enough to clear
    # 4.5:1 on them. A light-mode pill dropped onto a dark page is a bright
    # rectangle with dark text -- legible but shouting.
    "dark": dict(ok_bg="#16301a", ok_fg="#8fd694", ok_bd="#275230",
                 warn_bg="#332a12", warn_fg="#f0c265", warn_bd="#5c4a1d",
                 crit_bg="#3a1d1d", crit_fg="#f09a9a", crit_bd="#5e2e2e",
                 info_bg="#16273d", info_fg="#9cc4f5", info_bd="#27436b"),
}


def apply_theme() -> dict:
    """Resolve the theme once and publish it to the module globals."""
    global C_MODEL, C_MARKET, C_CLIM, SURFACE, INK, INK_2, INK_3, GRID, PANEL
    name = resolve_theme()
    P = dict(PALETTES[name])
    P.update(PILLS[name])
    P["name"] = name
    C_MODEL, C_MARKET, C_CLIM = P["model"], P["market"], P["clim"]
    SURFACE, INK, INK_2 = P["surface"], P["ink"], P["ink2"]
    INK_3, GRID, PANEL = P["ink3"], P["grid"], P["panel"]
    return P


def resolve_theme() -> str:
    """
    Which palette to paint with: the user's explicit choice, else whatever
    Streamlit is actually rendering in.

    `st.context.theme["type"]` reports the LIVE theme, including when it
    follows the operating system -- which a config file cannot tell you.
    """
    pick = st.session_state.get("theme_pick", "Match my system")
    if pick == "Light":
        return "light"
    if pick == "Dark":
        return "dark"
    try:
        return "dark" if st.context.theme.get("type") == "dark" else "light"
    except Exception:  # noqa: BLE001
        return "light"


def css(P: dict) -> str:
    return f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.015em; }}

  .hdr {{ display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
         margin-bottom:.15rem; }}
  .hdr h1 {{ margin:0; font-size:1.9rem; color:{P['ink']}; }}
  .hdr .wk {{ font-size:.95rem; color:{P['ink2']}; }}

  .strip {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.9rem 0 1.3rem; }}
  .pill {{ display:inline-flex; align-items:center; gap:.4rem;
          padding:.32rem .7rem; border-radius:999px; font-size:.8rem;
          font-weight:600; border:1px solid transparent; }}
  .pill.ok   {{ background:{P['ok_bg']};   color:{P['ok_fg']};
               border-color:{P['ok_bd']}; }}
  .pill.warn {{ background:{P['warn_bg']}; color:{P['warn_fg']};
               border-color:{P['warn_bd']}; }}
  .pill.crit {{ background:{P['crit_bg']}; color:{P['crit_fg']};
               border-color:{P['crit_bd']}; }}
  .pill.info {{ background:{P['info_bg']}; color:{P['info_fg']};
               border-color:{P['info_bd']}; }}

  .cards {{ display:grid; gap:.75rem;
           grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
           margin-bottom:1.1rem; }}
  .card {{ background:{P['surface']}; border:1px solid {P['grid']};
          border-radius:12px; padding:.8rem .95rem; }}
  .card .k {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
             color:{P['ink3']}; font-weight:700; }}
  .card .v {{ font-size:1.55rem; font-weight:700; color:{P['model']};
             line-height:1.25; font-variant-numeric:tabular-nums; }}
  .card .s {{ font-size:.76rem; color:{P['ink2']}; }}

  .lede {{ background:{P['panel']}; border-left:3px solid {P['model']};
          padding:.7rem .95rem; border-radius:0 8px 8px 0; font-size:.9rem;
          color:{P['ink2']}; margin-bottom:1rem; }}
  .lede b {{ color:{P['ink']}; }}

  .games {{ display:flex; gap:.45rem; flex-wrap:wrap; margin:.2rem 0 .4rem; }}
  .game {{ display:inline-flex; align-items:center; gap:.35rem;
          background:{P['surface']}; border:1px solid {P['grid']};
          border-radius:8px; padding:.28rem .55rem; font-size:.8rem;
          color:{P['ink2']}; }}
  .game b {{ color:{P['ink']}; font-weight:600; }}
  .game .t {{ color:{P['ink3']}; font-variant-numeric:tabular-nums; }}

  code {{ font-size:.85em; }}
  [data-testid="stMetricValue"] {{ font-size:1.4rem; }}
</style>
"""


# --- loading -----------------------------------------------------------

@st.cache_data(ttl=300)
def list_slates() -> list:
    return sorted(glob.glob(os.path.join(SLATE_DIR, "slate_*.parquet")), reverse=True)


@st.cache_data(ttl=300)
def slate_index() -> pd.DataFrame:
    """
    Every slate on disk with its season, week, and when its games kick off.

    ORDERING BY FILENAME IS A BUG, AND IT BIT. `list_slates` sorts reverse
    alphabetically, so a leftover `slate_2026_wk2.parquet` -- written on
    2026-09-11 by the old `current_week() + 1` logic, for games ten days
    away -- sorted ahead of week 1 and became the default. The dashboard
    opened on a week with no captured lines and reported "no posted lines,
    run run_slate.py" while week 1 sat right there, fully priced, one
    dropdown click away.

    A week is chosen by WHEN ITS GAMES ARE, which is the only thing that
    makes "current" mean anything.
    """
    rows = []
    for path in list_slates():
        season, week = season_week(path)
        if season is None:
            continue
        try:
            k = pl.read_parquet(path, columns=["kickoff"])["kickoff"]
            first, last = k.min(), k.max()
        except Exception:  # noqa: BLE001
            first = last = None
        rows.append({"path": path, "season": season, "week": week,
                     "first": first, "last": last,
                     "built": os.path.getmtime(path)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    df["first"] = pd.to_datetime(df["first"])
    df["last"] = pd.to_datetime(df["last"])
    return df


def default_slate(idx: pd.DataFrame) -> int:
    """
    The earliest week whose games have not all finished -- i.e. the one
    being played or coming next. Falls back to the most recent.
    """
    if idx.empty:
        return 0
    now = pd.Timestamp.utcnow().tz_localize(None)
    live = idx[idx["last"].notna() & (idx["last"] + pd.Timedelta(hours=3.5) > now)]
    return int(live.index[0]) if not live.empty else int(idx.index[-1])


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
                      unit: str, color: str = None, height: int = None):
    """
    Horizontal bars: magnitude by identity, which is what a leaderboard is.
    Every bar is directly labelled, so identity and value never depend on
    reading a colour or chasing an axis.

    `color=None` RATHER THAN `color=C_MODEL`. A default argument is bound
    once, when the function is defined -- which happens before
    `apply_theme()` has filled the palette in, so the default captured an
    empty string and every bar on the projections chart rendered invisible.
    The axis and the labels drew fine, which is what made it look like a
    data problem rather than a colour one.
    """
    color = color or C_MODEL
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

ET = "America/New_York"


def to_et(series) -> pd.Series:
    """
    Kickoffs as EASTERN wall-clock, which is how a game day is named.

    Not cosmetic. Sunday Night Football kicks off 20:20 Eastern, which is
    00:20 UTC the NEXT DAY, so grouping games by their UTC date files SNF
    under Monday and Monday Night Football under Tuesday. Anyone looking
    for "Sunday's games" would not find half of them.
    """
    t = pd.to_datetime(series, utc=True, errors="coerce")
    return t.dt.tz_convert(ET)


def day_options(slate: pd.DataFrame) -> pd.DataFrame:
    """One row per game day: the Eastern date, its games, and its status."""
    if "kickoff" not in slate.columns:
        return pd.DataFrame()
    k = slate.dropna(subset=["kickoff"]).copy()
    k["et"] = to_et(k["kickoff"])
    k["date"] = k["et"].dt.date
    now_et = pd.Timestamp.now(tz="UTC").tz_convert(ET)

    g = (k.groupby(["date", "game_id"])["et"].min().reset_index()
           .groupby("date")
           .agg(games=("game_id", "nunique"), first=("et", "min"),
                last=("et", "max"))
           .reset_index())
    # A day is done when its last game kicked off more than ~3.5 hours ago.
    g["done"] = (now_et - g["last"]).dt.total_seconds() / 3600.0 > 3.5
    g["label"] = g.apply(
        lambda r: f"{r['first']:%a %d %b} · {r['games']} game"
                  f"{'s' if r['games'] != 1 else ''}"
                  f"{'  ✓' if r['done'] else ''}", axis=1)
    return g.sort_values("date")


def filter_to_day(df: pd.DataFrame, day, time_col: str = "kickoff"):
    if day is None or time_col not in df.columns:
        return df
    d = df.dropna(subset=[time_col]).copy()
    return d[to_et(d[time_col]).dt.date == day]


def game_label(df: pd.DataFrame) -> pd.Series:
    """'AWAY @ HOME' from a game_id like 2026_01_NE_SEA."""
    return df["game_id"].str.split("_").map(
        lambda p: f"{p[2]} @ {p[3]}" if isinstance(p, list) and len(p) >= 4 else "")


# ---------------------------------------------------------------------
# 1. THE BOARD -- what you can actually bet on
# ---------------------------------------------------------------------

def view_board(slate: pd.DataFrame, season: int, week: int, day, day_name: str):
    """
    Only rows with a POSTED LINE. This is the page that answers "I want to
    bet on Sunday, who do I look at".

    WHY IT IS SEPARATE FROM PROJECTIONS, which was the single biggest
    source of confusion in the previous version: the slate holds 2,631
    projections across 632 players and six props, and **596 of them across
    157 players and two props have a line you can actually bet**. Showing
    them in one table meant scrolling past hundreds of unbettable rows to
    find the handful that matter. A projection for a player no book prices
    is a research output, not a betting one.
    """
    edges_file = os.path.join(CACHE_DIR, f"edges_{season}_wk{week}.parquet")
    edges = load_parquet(edges_file)
    if edges is None or edges.empty:
        # NAME THE WEEK AND SAY WHICH WEEKS DO HAVE LINES. The old message
        # just said "run run_slate.py", which is useless when the real
        # problem is that you are looking at the wrong week -- exactly what
        # happened when a stale week-2 slate became the default.
        have = []
        for f in glob.glob(os.path.join(CACHE_DIR, "edges_*.parquet")):
            m = re.search(r"edges_(\d+)_wk(\d+)\.parquet$", f)
            if m:
                have.append(f"{m.group(1)} week {m.group(2)}")
        st.warning(f"**No posted lines for {season} week {week}.**", icon="⚠️")
        if have:
            st.info(
                "Lines **are** captured for " + ", ".join(sorted(have)) +
                ".\n\nIf that is the week you meant, switch weeks in the "
                "sidebar on the left — this page follows whichever week is "
                "selected there.")
        else:
            st.info(
                "Run `python run_slate.py` — it buys the week's lines, "
                "rebuilds the slate and fills this page. Props post around "
                "the Wednesday before a week's games; before that the books "
                "return nothing, which is not an error.")
        return

    e = edges.copy()
    e["et"] = to_et(e["commence_time"])
    if day is not None:
        e = e[e["et"].dt.date == day]
    if e.empty:
        st.info(f"No posted lines for {day_name}.")
        return

    # attach the game each line belongs to, via the slate
    gid = (slate[["team", "game_id"]].drop_duplicates()
           if "game_id" in slate.columns else None)
    if gid is not None:
        e = e.merge(gid, on="team", how="left")
        e["game"] = game_label(e)
    else:
        e["game"] = ""

    e["side"] = np.where(e["edge"] >= 0, "OVER", "UNDER")
    e["gap"] = e["edge"].abs()
    e["kick"] = e["et"].dt.strftime("%a %-I:%M %p ET")

    big = int((e["gap"] >= 0.10).sum())
    cards([
        ("Bettable props", f"{len(e):,}", f"{day_name}"),
        ("Players priced", f"{e['player_id'].nunique()}",
         f"across {e['game'].nunique()} games"),
        ("We disagree ≥10 pts", f"{big}",
         "worth a look — and a bug check"),
        ("Book hold", f"{float(e['hold'].mean()):.1%}",
         "the vig, already removed"),
    ])

    st.markdown(
        '<div class="lede">'
        "<b>These are the only players with a line you can bet.</b> The "
        "slate projects everyone; the books price a fraction of them. Sort "
        "by <b>Gap</b> to see where the model and the book disagree most — "
        "but read those as places to check for a bug first, since the model "
        "has no demonstrated edge over a closing line yet."
        "</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([3, 2, 2])
    games = sorted(g for g in e["game"].unique() if g)
    pick_games = f1.multiselect("Game", games, placeholder="All games")
    props = sorted(e["prop"].unique())
    pick_prop = f2.multiselect("Prop", props, format_func=fmt,
                               placeholder="All props")
    min_books = f3.slider("Minimum books quoting", 1, 5, 2,
                          help="A single-book line is much weaker evidence "
                               "than a five-book consensus.")

    v = e[e["n_books"] >= min_books]
    if pick_games:
        v = v[v["game"].isin(pick_games)]
    if pick_prop:
        v = v[v["prop"].isin(pick_prop)]
    if v.empty:
        st.info("Nothing matches those filters.")
        return

    v = v.sort_values("gap", ascending=False).copy()
    v["prop_label"] = v["prop"].map(fmt)
    if "position" in v.columns:
        v["pos"] = v["position"].map(position_tag)
    # PERCENTAGE POINTS, NOT FRACTIONS. Streamlit's ProgressColumn applies
    # the format string to the RAW value, so a 0-1 probability with
    # "%.0f%%" renders every row as "0%" -- which is what the first version
    # of this table did, for all 546 rows.
    for c in ("model_prob", "market_prob", "gap"):
        v[c] = v[c] * 100.0

    st.dataframe(
        v[["matched_name", "prop_label", "line", "side", "gap",
           "model_prob", "market_prob", "n_books", "game", "kick",
           "availability"]],
        use_container_width=True, hide_index=True, height=560,
        column_config={
            "matched_name": st.column_config.TextColumn("Player", width="medium"),
            "prop_label": st.column_config.TextColumn("Prop", width="small"),
            "line": st.column_config.NumberColumn("Line", format="%.1f",
                                                  width="small"),
            "side": st.column_config.TextColumn(
                "We lean", width="small",
                help="Which side of the line the model prefers"),
            "gap": st.column_config.NumberColumn(
                "Gap", format="%.0f pts", width="small",
                help="How far apart the model and the book are, in "
                     "percentage points. Sort by this."),
            "model_prob": st.column_config.ProgressColumn(
                "Model P(over)", min_value=0.0, max_value=100.0,
                format="%.0f%%"),
            "market_prob": st.column_config.ProgressColumn(
                "Book P(over)", min_value=0.0, max_value=100.0,
                format="%.0f%%"),
            "n_books": st.column_config.NumberColumn("Books", format="%d",
                                                     width="small"),
            "game": st.column_config.TextColumn("Game", width="small"),
            "kick": st.column_config.TextColumn("Kickoff", width="small"),
            "availability": st.column_config.TextColumn("Status", width="small"),
        })
    st.caption(
        "**Model P(over)** is the model's probability the player goes over "
        "that line. **Book P(over)** is the same thing implied by the price, "
        "with the vig removed. **Gap** is the distance between them, in "
        "percentage points — click the column header to sort.")


# ---------------------------------------------------------------------
# 2. PROJECTIONS -- everyone, priced or not
# ---------------------------------------------------------------------

def view_projections(slate: pd.DataFrame, season: int, week: int, day,
                     day_name: str):
    df = filter_to_day(slate, day)
    if df.empty:
        st.info(f"No projections for {day_name}.")
        return

    edges = load_parquet(os.path.join(CACHE_DIR, f"edges_{season}_wk{week}.parquet"))
    priced = set() if edges is None or edges.empty else set(edges["player_id"])

    st.markdown(
        '<div class="lede">'
        "Every player the model projects, whether a book prices him or not. "
        "Use this to look someone up. For what you can actually bet, use the "
        "<b>Betting Board</b> tab."
        "</div>", unsafe_allow_html=True)

    props = [p for p in PROP_LABEL if p in set(df["prop"])]
    props += [p for p in sorted(set(df["prop"])) if p not in props]

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Top projections")
        c1, c2 = st.columns([3, 2])
        prop = c1.selectbox("Prop", props, format_func=fmt, key="lb_prop")
        topn = c2.slider("Show", 5, 30, 12, key="lb_n")
        only_priced = st.checkbox(
            "Only players with a posted line", value=False,
            help="Narrows to the players a book actually prices.")
        sub = df[df["prop"] == prop]
        if only_priced and priced:
            sub = sub[sub["player_id"].isin(priced)]
        top = sub.nlargest(topn, "expected")
        if top.empty:
            st.info("Nothing projected here.")
        else:
            st.altair_chart(
                leaderboard_chart(top, "expected", fmt(prop),
                                  PROP_UNIT.get(prop, "")),
                use_container_width=True)
            st.caption(
                f"The model's **average** outcome, not its most likely one — "
                f"these distributions are right-skewed, so the median sits "
                f"below the mean.")

    with right:
        st.markdown("#### Most likely to score")
        td = load_parquet(os.path.join(
            SLATE_DIR, f"anytime_td_{season}_wk{week}.parquet"))
        if td is None or td.empty:
            st.info("No anytime-TD file for this slate.")
        else:
            teams = set(df["team"])
            t = td[td["team"].isin(teams)].nlargest(12, "anytime_td")
            if t.empty:
                st.info("No TD projections for these teams.")
            else:
                st.altair_chart(
                    leaderboard_chart(t, "anytime_td", "P(scores a TD)", "TD",
                                      color=C_CLIM),
                    use_container_width=True)
                st.caption(
                    "P(anytime) = 1 − P(no receiving TD) × P(no rushing TD), "
                    "treated as independent — which slightly **understates** "
                    "goal-line backs.")

    st.divider()
    st.markdown("#### Look up a player")
    f1, f2, f3 = st.columns([2, 2, 3])
    fprop = f1.multiselect("Prop", props, format_func=fmt, key="tbl_prop",
                           placeholder="All props")
    fteam = f2.multiselect("Team", sorted(df["team"].unique()), key="tbl_team",
                           placeholder="All teams")
    search = f3.text_input("Name contains", key="tbl_q", placeholder="e.g. Nacua")

    t = df
    if fprop:
        t = t[t["prop"].isin(fprop)]
    if fteam:
        t = t[t["team"].isin(fteam)]
    if search:
        t = t[t["player_name"].str.contains(search, case=False, na=False)]

    show = t.copy()
    show["prop"] = show["prop"].map(fmt)
    show["bet"] = np.where(show["player_id"].isin(priced), "✓", "")
    if "position" in show.columns:
        show["position"] = show["position"].map(position_tag)
    if "kickoff" in show.columns:
        show["kicks"] = to_et(show["kickoff"]).dt.strftime("%a %-I:%M %p")
    cols = [c for c in ["player_name", "team", "kicks", "position", "prop",
                        "expected", "bet", "availability"] if c in show.columns]
    show = show.sort_values("expected", ascending=False)[cols]
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=420,
        column_config={
            "player_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "kicks": st.column_config.TextColumn("Kickoff", width="small",
                                                 help="Eastern"),
            "position": st.column_config.TextColumn("Pos", width="small"),
            "prop": st.column_config.TextColumn("Prop", width="medium"),
            "expected": st.column_config.NumberColumn("Projected", format="%.1f"),
            "bet": st.column_config.TextColumn(
                "Line?", width="small",
                help="✓ means a book posted a line — see the Betting Board"),
            "availability": st.column_config.TextColumn("Status", width="small"),
        })


# ---------------------------------------------------------------------
# The day strip
# ---------------------------------------------------------------------

def day_header(slate: pd.DataFrame, days: pd.DataFrame, day):
    """The chosen day's games, with kickoff times and whether they are done."""
    if "kickoff" not in slate.columns or day is None:
        return
    k = slate.dropna(subset=["kickoff"]).copy()
    k["et"] = to_et(k["kickoff"])
    k = k[k["et"].dt.date == day]
    if k.empty:
        return
    now_et = pd.Timestamp.now(tz="UTC").tz_convert(ET)

    g = (k.groupby("game_id")
           .agg(et=("et", "min"), players=("player_id", "nunique"))
           .reset_index())
    g["game"] = game_label(g)
    g["mins"] = (now_et - g["et"]).dt.total_seconds() / 60.0
    g["status"] = np.where(g["mins"] > 210, "Final",
                  np.where(g["mins"] >= 0, "Playing now", "Upcoming"))
    g["when"] = g["et"].dt.strftime("%-I:%M %p ET")
    g = g.sort_values("et")

    ICON = {"Final": "✓", "Playing now": "●", "Upcoming": "○"}
    chips = "".join(
        f'<span class="game">{ICON[r["status"]]} <b>{r["game"]}</b> '
        f'<span class="t">{r["when"]}</span></span>'
        for r in g.to_dict("records"))
    st.markdown(f'<div class="games">{chips}</div>', unsafe_allow_html=True)

    upcoming = g[g["status"] == "Upcoming"]
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        hrs = -nxt["mins"] / 60.0
        st.caption(f"Next kickoff: **{nxt['game']}** in {hrs:.0f} hours "
                   f"({nxt['when']}).  ✓ final · ● playing now · ○ upcoming")
    else:
        st.caption("Every game on this day has kicked off.  "
                   "✓ final · ● playing now · ○ upcoming")


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
    P = apply_theme()
    st.markdown(css(P), unsafe_allow_html=True)

    slates = list_slates()
    if not slates:
        st.title("🏈 Football Props")
        st.info("No slate yet. Run **`python run_slate.py`** to build one.")
        return

    idx = slate_index()
    now = pd.Timestamp.utcnow().tz_localize(None)

    def week_label(i: int) -> str:
        r = idx.loc[i]
        when = ""
        if pd.notna(r["first"]):
            lo = to_et(pd.Series([r["first"]])).iloc[0]
            hi = to_et(pd.Series([r["last"]])).iloc[0]
            when = f" · {lo:%d %b}–{hi:%d %b}"
        tag = ""
        if pd.notna(r["last"]):
            if r["last"] + pd.Timedelta(hours=3.5) < now:
                tag = "  ✓ done"
            elif pd.notna(r["first"]) and r["first"] < now:
                tag = "  ● in progress"
        return f"Week {r['week']}{when}{tag}"

    with st.sidebar:
        st.radio("Appearance", ["Match my system", "Light", "Dark"],
                 key="theme_pick", horizontal=False,
                 help="Charts and text are re-coloured for the surface they "
                      "sit on, not flipped.")
        st.divider()
        st.markdown("### Week")
        choice = st.selectbox(
            "Week", list(idx.index), label_visibility="collapsed",
            index=default_slate(idx), format_func=week_label)
        path = idx.loc[choice, "path"]
        st.caption(
            "An NFL week runs Thursday to Monday, so one week holds several "
            "game days. Weeks are ordered by **when the games are**, not by "
            "filename, and the one being played is selected by default.")

        # SELF-DIAGNOSIS. "It says there is nothing there" is unanswerable
        # without knowing which paths were checked and what was found, and
        # that question has now come up twice.
        with st.expander("Where is it reading from?"):
            st.caption(f"**Project**  \n`{BASE_DIR}`")
            st.caption(f"**Launched from**  \n`{os.getcwd()}`")
            n_sl = len(glob.glob(os.path.join(SLATE_DIR, "slate_*.parquet")))
            n_ed = len(glob.glob(os.path.join(CACHE_DIR, "edges_*.parquet")))
            ok_log = os.path.exists(os.path.join(CACHE_DIR,
                                                 "scoring_log.parquet"))
            st.caption(
                f"slates/ → **{n_sl}** slate file{'s' if n_sl != 1 else ''}  \n"
                f"cache/ → **{n_ed}** edges file{'s' if n_ed != 1 else ''}, "
                f"scoring log {'found' if ok_log else '**missing**'}")
            if n_sl and not n_ed:
                st.caption(
                    "Slates but no edges means the capture step has not run "
                    "for any week, or it ran and bought nothing.")

        # A slate for a week whose games are still days away, built before
        # its props would even have posted, is a leftover -- usually from an
        # aborted run. Say so rather than letting it look current.
        r = idx.loc[choice]
        if pd.notna(r["first"]) and r["first"] > now + pd.Timedelta(days=3):
            st.warning(
                f"Week {r['week']}'s first game is "
                f"{(r['first'] - now).days} days away. Books post props "
                f"around the Wednesday before, so this slate is early and "
                f"will have no lines yet.", icon="⚠️")

    slate = load_parquet(path)
    season, week = season_week(path)
    if slate is None or slate.empty:
        st.warning("Empty slate.")
        return

    st.markdown(
        f'<div class="hdr"><h1>🏈 Football Props</h1>'
        f'<span class="wk">{season} · Week {week}</span></div>',
        unsafe_allow_html=True)

    # THE DAY PICKER IS THE PRIMARY CONTROL. "I want to bet on Sunday" is
    # the question this dashboard exists to answer, and in the previous
    # version there was no way to ask it -- the whole week was one
    # undifferentiated pile of 2,631 rows.
    days = day_options(slate)
    day, day_name = None, f"week {week}"
    if not days.empty:
        labels = ["Whole week"] + days["label"].tolist()
        default = 0
        live = days[~days["done"]]
        if not live.empty:
            default = int(days.index.get_loc(live.index[0])) + 1
        choice = st.radio("Game day", labels, index=default, horizontal=True,
                          label_visibility="collapsed")
        if choice != "Whole week":
            row = days[days["label"] == choice].iloc[0]
            day, day_name = row["date"], choice.split(" ·")[0]

    edges_path = os.path.join(CACHE_DIR, f"edges_{season}_wk{week}.parquet")
    log = load_parquet(os.path.join(CACHE_DIR, "scoring_log.parquet"))
    n_weeks = 0 if log is None or log.empty else len(
        log[["season", "week"]].drop_duplicates())
    st.markdown(
        '<div class="strip">'
        + pill("crit", "◆ No proven edge over the book — read this as research")
        + pill("info", f"◆ {n_weeks} week{'s' if n_weeks != 1 else ''} scored "
                       f"of ~12–15 needed")
        + pill("ok" if os.path.exists(edges_path) else "warn",
               ("◆ Lines captured" if os.path.exists(edges_path)
                else "◆ No lines captured this week"))
        + "</div>", unsafe_allow_html=True)

    day_header(slate, days, day)

    t1, t2, t3, t4 = st.tabs(
        ["🎯 Betting Board", "📋 All Projections", "📈 Track Record",
         "🩺 Model Health"])
    with t1:
        view_board(slate, season, week, day, day_name)
    with t2:
        view_projections(slate, season, week, day, day_name)
    with t3:
        view_record()
    with t4:
        view_health()


if __name__ == "__main__":
    main()
