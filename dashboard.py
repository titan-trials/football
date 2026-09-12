"""
football_props dashboard.

    streamlit run dashboard.py

Four tabs, each answering one question:

    Slate        what does the model say about the upcoming week?
    Edges        where does it disagree with the market, and by how much?
    Scoring      is it calibrated, and does it beat the null?
    Flags        has the availability flag earned its way into the model yet?

DESIGN NOTE
-----------
Most of this is tables, deliberately. A slate is an identity-and-lookup
problem -- "what does it say about Puka Nacua" -- and a table answers that
better than any chart. Charts appear only where the question is genuinely
about shape: calibration (a line against the identity diagonal) and edge
distribution.

The three-colour categorical palette below passed all six checks of the
dataviz validator in light mode (worst adjacent CVD dE 22.9, normal 27.4),
so model/market/climatology stay distinguishable for colourblind readers.
Series are also always direct-labelled, so identity is never colour-alone.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

st.set_page_config(page_title="football_props", layout="wide")

# Validated categorical palette: model / market / climatology.
C_MODEL, C_MARKET, C_CLIM = "#2f6fb5", "#d1762a", "#7b57b8"
SURFACE = "#fcfcfb"

SLATE_DIR = "slates"
CACHE_DIR = "cache"


# --- loading -----------------------------------------------------------

@st.cache_data(ttl=300)
def list_slates() -> list:
    return sorted(glob.glob(os.path.join(SLATE_DIR, "slate_*.parquet")), reverse=True)


@st.cache_data(ttl=300)
def load_parquet(path: str):
    return pl.read_parquet(path).to_pandas() if os.path.exists(path) else None


def line_cols(df) -> list:
    return [c for c in df.columns if c.startswith("over_")]


# --- tabs --------------------------------------------------------------

def tab_slate():
    slates = list_slates()
    if not slates:
        st.info("No slates yet. Run `python predict_slate.py`.")
        return
    path = st.selectbox("Slate", slates, format_func=os.path.basename)
    df = load_parquet(path)
    if df is None or df.empty:
        st.warning("Empty slate.")
        return

    stamp = df["predicted_at"].max() if "predicted_at" in df else None
    n_clean = int(df["clean"].sum()) if "clean" in df else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rows", f"{len(df):,}")
    c2.metric("players", f"{df['player_id'].nunique():,}")
    c3.metric("clean", f"{n_clean:,} / {len(df):,}",
              help="Predicted before that row's own kickoff. Clean is a property "
                   "of the row, not the slate — five game windows a week.")
    c4.metric("written", str(stamp)[:16] if stamp is not None else "—")

    if "availability" in df and (df["availability"] == "CLEAR").all():
        st.warning(
            "Every row is CLEAR. If the week's injury report has not published "
            "yet, that means the report is ABSENT — not that players are healthy. "
            "Re-run after Wednesday.")

    f1, f2, f3 = st.columns([2, 2, 3])
    prop = f1.selectbox("Prop", sorted(df["prop"].unique()))
    teams = f2.multiselect("Team", sorted(df["team"].unique()))
    sub = df[df["prop"] == prop]
    if teams:
        sub = sub[sub["team"].isin(teams)]
    search = f3.text_input("Player contains")
    if search:
        sub = sub[sub["player_name"].str.contains(search, case=False, na=False)]

    cols = (["player_name", "team", "position", "availability", "exp_opportunities",
             "share", "shape", "expected"] + line_cols(sub))
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(
        sub.sort_values("expected", ascending=False)[cols],
        use_container_width=True, hide_index=True, height=520)

    td_path = path.replace("slate_", "anytime_td_")
    td = load_parquet(td_path)
    if td is not None and not td.empty:
        st.subheader("Anytime touchdown")
        st.caption(
            "P(anytime) = 1 − P(no receiving TD) × P(no rushing TD). The two are "
            "assumed independent given opportunity counts; they are positively "
            "correlated for goal-line backs, so this slightly understates them "
            "for exactly the players the market prices most sharply.")
        st.dataframe(td.sort_values("anytime_td", ascending=False).head(40),
                     use_container_width=True, hide_index=True)


def tab_edges():
    mk = load_parquet(os.path.join(CACHE_DIR, "market_log.parquet"))
    if mk is None or mk.empty:
        st.info(
            "No market log yet. Run `python -c \"from data.odds import "
            "fetch_slate_props, append_market_log; r,c,_ = fetch_slate_props("
            "dry_run=True)\"` to check the cost first, then drop dry_run.\n\n"
            "Player props cost ~16 credits per market per week live, and "
            "**10× that historically** — so a week not captured is a week that "
            "can only be bought back at ten times the price.")
        return

    slates = list_slates()
    if not slates:
        st.info("No slate to compare against.")
        return
    df = load_parquet(slates[0])
    st.caption(f"Market log: {len(mk):,} rows. Slate: {os.path.basename(slates[0])}")
    st.dataframe(mk.sort_values("captured_at", ascending=False).head(200),
                 use_container_width=True, hide_index=True)
    st.info(
        "Edge = model probability − de-vigged market probability. Joining "
        "player names between the books and nflverse ids is the unsolved part; "
        "do it by fuzzy name within team before trusting any edge shown here.")


def tab_scoring():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "compare_*.parquet")))
    if not files:
        st.info("No scored files yet. Run `python compare_props.py`.")
        return
    path = st.selectbox("Scored prop", files, format_func=os.path.basename)
    df = load_parquet(path)
    if df is None or df.empty:
        return

    pcols = [c for c in df.columns if c.startswith("p_")]
    if not pcols:
        st.warning("No probability columns.")
        return
    line = st.selectbox("Line", pcols, format_func=lambda c: c[2:])
    L = float(line[2:])
    ccol = "c_" + line[2:]

    y = (df["actual"] > L).astype(float).values
    p = df[line].values

    def brier_skill(y, p):
        b = float(np.mean(y)); d = b * (1 - b)
        return np.nan if d <= 0 else 1 - float(np.mean((p - y) ** 2)) / d

    c1, c2, c3 = st.columns(3)
    c1.metric("rows", f"{len(df):,}")
    c2.metric("base rate", f"{y.mean():.3f}")
    c3.metric("model Brier skill", f"{brier_skill(y, p):+.4f}",
              help="Pooled, never averaged per week — skill is 1 − brier/(p(1−p)) "
                   "and the average of a ratio is not the ratio of averages.")

    # Calibration: predicted vs observed, against the identity diagonal.
    q = np.quantile(p, np.linspace(0, 1, 9))
    rows = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        rows.append({"predicted": float(p[m].mean()),
                     "observed": float(y[m].mean()), "n": int(m.sum())})
    if rows:
        import pandas as pd
        cal = pd.DataFrame(rows)
        base = alt.Chart(cal).encode(
            x=alt.X("predicted:Q", title="predicted probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed:Q", title="observed frequency",
                    scale=alt.Scale(domain=[0, 1])))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#9a9a95", strokeWidth=1).encode(x="x:Q", y="x:Q")
        pts = base.mark_point(size=90, filled=True, color=C_MODEL,
                              stroke=SURFACE, strokeWidth=2).encode(
            tooltip=["predicted", "observed", "n"])
        ln = base.mark_line(color=C_MODEL, strokeWidth=2)
        st.altair_chart((diag + ln + pts).properties(height=340),
                        use_container_width=True)
        st.caption(
            "Points on the dashed diagonal are perfectly calibrated. Below it "
            "means over-confident. One series, so no legend — the title names it.")
        st.dataframe(cal, use_container_width=True, hide_index=True)

    if ccol in df.columns:
        st.metric("climatology Brier skill", f"{brier_skill(y, df[ccol].values):+.4f}",
                  help="The honest null: the player's own trailing distribution. "
                       "Beating it is necessary, not impressive. The market is the "
                       "bar that matters.")


def tab_flags():
    st.subheader("Availability flag — logged, fed to nothing")
    st.markdown(
        "The flag is a column on every slate and an input to no model. After "
        "~20 scored weeks, `data.availability.self_test` reports whether "
        "flagged players underperformed their predictions. If they did, wire "
        "it in. If not, delete the column. Nothing was ever at risk.\n\n"
        "**What the evidence says so far** — from the Stage 1 top-bucket "
        "investigation on 2025, and it is thin:"
    )
    import pandas as pd
    st.dataframe(pd.DataFrame([
        {"bucket": "no report listed", "n": 776, "predicted": 0.774,
         "actual": 0.705, "gap": -0.069},
        {"bucket": "Questionable", "n": 47, "predicted": 0.808,
         "actual": 0.660, "gap": -0.149},
    ]), use_container_width=True, hide_index=True)
    st.caption(
        "Flagged players miss badly, but they were 5.7% of the bucket — removing "
        "them moved the aggregate gap only from −0.073 to −0.069. Real "
        "per-player, negligible in aggregate, n = 47. Not yet evidence.")

    slates = list_slates()
    if slates:
        df = load_parquet(slates[0])
        if df is not None and "availability" in df:
            counts = (df.groupby("availability")["player_id"].nunique()
                        .reset_index(name="players"))
            st.dataframe(counts, use_container_width=True, hide_index=True)


def main():
    st.title("football_props")
    st.caption(
        "NFL player-prop model. Two stages: opportunity (targets/carries/attempts) "
        "then outcome per opportunity, compounded by exact convolution. Sibling to "
        "baseball_predictor — see CONTEXT.md for what has been tried and what it "
        "measured."
    )
    t1, t2, t3, t4 = st.tabs(["Slate", "Edges", "Scoring", "Flags"])
    with t1:
        tab_slate()
    with t2:
        tab_edges()
    with t3:
        tab_scoring()
    with t4:
        tab_flags()


if __name__ == "__main__":
    main()
