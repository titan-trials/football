"""
Slice the served-population backtest by depth-chart role.

    python validate_population.py
    python validate_population.py --tag _roster --compare _statsfit

WHY THIS SCRIPT EXISTS
----------------------
`compare_props.py` now scores the population `predict_slate` serves: the
depth chart minus the players who are known days ahead not to be playing.
That doubled the row count, because a listed, available WR6 who caught
nothing is a row with actual 0.

Those zero rows are the right rows -- dropping them is how a model that
under-predicts everybody still measures unbiased -- but they make the
headline skill number hard to read. Half the sample is nearly free to
predict, so skill against climatology went UP while nothing about the model
changed. The honest question is whether the model is calibrated for the
players a book would actually post a line on.

HOW THE SLICE IS CHOSEN
-----------------------
By `role_bucket`, the depth-chart rank, which is **known before kickoff**.

The two obvious alternatives are both mistakes I have already made in this
project and written down:

  * slicing by the PREDICTION hides compression -- every bin looks
    calibrated when the model systematically shrinks, because the bin
    boundaries move with the shrinkage;
  * slicing by the OUTCOME (say, "players who recorded at least one
    target") conditions on success and guarantees the model looks low.

Role rank is available at predict time, so a per-bucket bias is a bias the
model could in principle be corrected for.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from features.props import PROPS
from model.scoring import brier_skill

CACHE = "cache"


def load(prop: str, tag: str) -> pl.DataFrame | None:
    p = os.path.join(CACHE, f"compare_{prop}{tag}.parquet")
    return pl.read_parquet(p) if os.path.exists(p) else None


def by_role(df: pl.DataFrame) -> pl.DataFrame:
    return (df.group_by(["role_pos", "role_bucket"])
              .agg([pl.len().alias("n"),
                    pl.col("actual").mean().alias("mean_actual"),
                    pl.col("exp").mean().alias("mean_pred"),
                    pl.col("crps_model").mean().alias("crps"),
                    pl.col("crps_clim").mean().alias("crps_clim"),
                    (pl.col("actual") > 0).mean().alias("share_nonzero")])
              .with_columns([
                  (pl.col("mean_pred") - pl.col("mean_actual")).alias("bias"),
                  pl.when(pl.col("mean_actual") > 0)
                    .then(pl.col("mean_pred") / pl.col("mean_actual"))
                    .otherwise(None).alias("ratio"),
                  (1 - pl.col("crps") / pl.col("crps_clim")).alias("skill"),
              ])
              .sort(["role_pos", "role_bucket"]))


def contributor_view(prop: str, df: pl.DataFrame, top: int) -> None:
    """
    The model restricted to the top `top` depth-chart ranks at each
    position -- roughly the set a book posts lines on.
    """
    spec = PROPS[prop]
    sub = df.filter(pl.col("role_bucket") <= top)
    if sub.is_empty():
        return
    mp, ma = float(sub["exp"].mean()), float(sub["actual"].mean())
    print(f"\n  role_bucket <= {top}:  n={sub.height}  pred {mp:.2f}  "
          f"actual {ma:.2f}  ratio {mp / ma if ma else float('nan'):.3f}  "
          f"CRPS {float(sub['crps_model'].mean()):.3f}")
    for L in spec.lines:
        col = f"p_{L}"
        if col not in sub.columns:
            continue
        y = (sub["actual"] > L).to_numpy().astype(float)
        if y.mean() <= 0.02 or y.mean() >= 0.98:
            continue
        p = sub[col].to_numpy()
        print(f"      over {L:7}  base {y.mean():.3f}  mean p {p.mean():.3f}  "
              f"skill {brier_skill(y, p):+.4f}  "
              f"clim {brier_skill(y, sub[f'c_{L}'].to_numpy()):+.4f}")


def main(tag: str, compare: str, top: int, props: list[str]) -> None:
    for prop in props:
        df = load(prop, tag)
        if df is None:
            continue
        print(f"\n{'=' * 78}\n{prop}   n={df.height}   ({tag or 'no tag'})\n{'=' * 78}")
        t = by_role(df)
        print(f"{'role':6} {'rk':>3} {'n':>6} {'actual':>8} {'pred':>8} "
              f"{'ratio':>7} {'nonzero':>8} {'CRPS':>8} {'skill':>8}")
        for r in t.iter_rows(named=True):
            if r["n"] < 25:
                continue
            ratio = f"{r['ratio']:7.3f}" if r["ratio"] is not None else "      -"
            print(f"{str(r['role_pos']):6} {r['role_bucket']:3} {r['n']:6} "
                  f"{r['mean_actual']:8.2f} {r['mean_pred']:8.2f} {ratio} "
                  f"{r['share_nonzero']:8.3f} {r['crps']:8.3f} {r['skill']:+8.4f}")
        contributor_view(prop, df, top)

        if compare:
            other = load(prop, compare)
            if other is None:
                continue
            # A/B ON IDENTICAL ROWS. Anything else compares two samples as
            # well as two models, and then the difference means nothing.
            j = df.join(other.select(["week", "team", "player_id", "exp", "crps_model"]),
                        on=["week", "team", "player_id"], how="inner", suffix="_b")
            if j.is_empty():
                continue
            print(f"\n  A/B vs {compare} on {j.height} identical rows")
            print(f"    CRPS   {tag or 'base'} {float(j['crps_model'].mean()):.4f}   "
                  f"{compare} {float(j['crps_model_b'].mean()):.4f}")
            print(f"    bias   {tag or 'base'} "
                  f"{float(j['exp'].mean() - j['actual'].mean()):+.3f}   "
                  f"{compare} {float(j['exp_b'].mean() - j['actual'].mean()):+.3f}")
            for L in PROPS[prop].lines:
                y = (j["actual"] > L).to_numpy().astype(float)
                if y.mean() <= 0.02 or y.mean() >= 0.98 or f"p_{L}" not in j.columns:
                    continue
                print(f"    over {L:7} skill {tag or 'base'} "
                      f"{brier_skill(y, j[f'p_{L}'].to_numpy()):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--props", nargs="+", default=list(PROPS))
    a = ap.parse_args()
    main(a.tag, a.compare, a.top, a.props)
