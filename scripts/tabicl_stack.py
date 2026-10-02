"""TabICL v2 as a third forecaster, walk-forward, on top of the backtest output.

TabICL is a tabular foundation model: it predicts from (rows, labels) in one
forward pass with no training or tuning. Here it is given only information
available before each block, and only matches already forecast out of sample.

Two versions:
* tab   : features are the classical and Bayesian model probabilities plus
          promoted flags and an early-season indicator (no market data).
* tabm  : the same plus the de-margined closing market probabilities, to test
          whether it can combine models and market better than the market alone.

Run with the .venv-tabicl interpreter:
    .venv-tabicl/bin/python scripts/tabicl_stack.py [base|norm]
Season 2021-22 is history only (nothing earlier was forecast out of sample).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier

RES = Path(__file__).resolve().parent.parent / "results"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
SUFFIX = "" if VARIANT == "base" else f"_{VARIANT}"
BLOCK_DAYS = 28
MODEL_COLS = ["mle_h", "mle_d", "mle_a", "bayes_h", "bayes_d", "bayes_a", "home_promoted", "away_promoted", "early"]
MARKET_COLS = ["mkt_h", "mkt_d", "mkt_a"]


def outcome(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def main():
    df = pd.read_csv(RES / f"predictions{SUFFIX}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["y"] = outcome(df.hg.to_numpy(), df.ag.to_numpy())
    df["early"] = (df.groupby("season").cumcount() < 100).astype(float)
    df[["home_promoted", "away_promoted"]] = df[["home_promoted", "away_promoted"]].fillna(0.0)
    for c in ["tab_h", "tab_d", "tab_a", "tabm_h", "tabm_d", "tabm_a"]:
        df[c] = np.nan

    t0 = time.time()
    for season in sorted(df.season.unique())[1:]:
        sea = df[df.season == season]
        start, end = sea.date.min(), sea.date.max()
        edges = list(pd.date_range(start, end + pd.Timedelta(days=1), freq=f"{BLOCK_DAYS}D")) + [end + pd.Timedelta(days=1)]
        for b0, b1 in zip(edges, edges[1:]):
            test = (df.season == season) & (df.date >= b0) & (df.date < b1)
            if not test.any():
                continue
            train = df.date < b0
            for name, cols in [("tab", MODEL_COLS), ("tabm", MODEL_COLS + MARKET_COLS)]:
                clf = TabICLClassifier(random_state=0, device="cpu")
                clf.fit(df.loc[train, cols].to_numpy(float), df.loc[train, "y"].to_numpy())
                p = clf.predict_proba(df.loc[test, cols].to_numpy(float))
                # align columns to classes 0, 1, 2
                full = np.zeros((len(p), 3))
                full[:, clf.classes_] = p
                df.loc[test, [f"{name}_h", f"{name}_d", f"{name}_a"]] = full
            print(f"[{time.time() - t0:6.0f}s] {season} {b0.date()} train={int(train.sum())} test={int(test.sum())}", flush=True)
            df.to_csv(RES / f"predictions_tabicl{SUFFIX}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
