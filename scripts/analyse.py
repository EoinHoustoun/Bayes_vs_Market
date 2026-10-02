"""Score the backtest and write results/summary.json plus tables.

Questions answered:
1. How do the classical (MLE) and Bayesian models score against the closing market?
2. Is the Bayesian model better than the classical one, with a bootstrap interval?
3. Where does it help: promoted sides, early-season blocks, draws?
4. Does either model add information to the market (walk-forward log-linear pool)?
5. Are the draw probabilities calibrated?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bf import metrics  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
MODELS = ["mkt", "mle", "bayes"]


def probs(df, m):
    return df[[f"{m}_h", f"{m}_d", f"{m}_a"]].to_numpy()


def pool(p_model, p_mkt, w):
    """Log-linear pool: market with weight 1 - w, model with weight w."""
    lp = (1 - w) * np.log(np.clip(p_mkt, 1e-12, 1)) + w * np.log(np.clip(p_model, 1e-12, 1))
    e = np.exp(lp)
    return e / e.sum(axis=1, keepdims=True)


def main():
    df = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    df["y"] = metrics.outcome(df.hg, df.ag)
    y = df.y.to_numpy()
    out = {"n_matches": int(len(df)), "seasons": sorted(df.season.unique())}

    # 1. overall and per-season scores
    out["overall"] = {m: metrics.score_all(probs(df, m), y) for m in MODELS}
    out["by_season"] = {s: {m: metrics.score_all(probs(g, m), g.y.to_numpy()) for m in MODELS}
                        for s, g in df.groupby("season")}

    # 2. paired bootstrap on per-match log loss
    ll = {m: metrics.per_match_log_loss(probs(df, m), y) for m in MODELS}
    out["diff_bayes_minus_mle"] = metrics.bootstrap_diff(ll["bayes"], ll["mle"])
    out["diff_bayes_minus_mkt"] = metrics.bootstrap_diff(ll["bayes"], ll["mkt"])
    out["diff_mle_minus_mkt"] = metrics.bootstrap_diff(ll["mle"], ll["mkt"])

    # 3. subsets
    prom = (df.home_promoted == 1) | (df.away_promoted == 1)
    df["season_match_no"] = df.groupby("season").cumcount()
    early = df.season_match_no < 100   # roughly the first 10 gameweeks
    subsets = {"promoted_involved": prom, "promoted_early": prom & early, "early_season": early,
               "rest": ~early}
    out["subsets"] = {}
    for name, m in subsets.items():
        g = df[m]
        out["subsets"][name] = {"n": int(m.sum()), **{k: metrics.log_loss(probs(g, k), g.y.to_numpy()) for k in MODELS},
                                "bayes_minus_mle": metrics.bootstrap_diff(ll["bayes"][m.to_numpy()], ll["mle"][m.to_numpy()])}

    # 4. walk-forward pooling: choose w on all earlier seasons, apply to the next
    seasons = sorted(df.season.unique())
    grid = np.round(np.arange(0, 0.61, 0.05), 2)
    out["pool"] = {}
    for model in ["mle", "bayes"]:
        rows, pooled_ll, base_ll = [], [], []
        for i, s in enumerate(seasons[1:], start=1):
            past = df[df.season.isin(seasons[:i])]
            best = min(grid, key=lambda w: metrics.log_loss(pool(probs(past, model), probs(past, "mkt"), w), past.y.to_numpy()))
            cur = df[df.season == s]
            p = pool(probs(cur, model), probs(cur, "mkt"), best)
            pooled_ll.append(metrics.per_match_log_loss(p, cur.y.to_numpy()))
            base_ll.append(metrics.per_match_log_loss(probs(cur, "mkt"), cur.y.to_numpy()))
            rows.append({"season": s, "w": float(best)})
        out["pool"][model] = {"weights": rows,
                              "pooled_minus_market": metrics.bootstrap_diff(np.concatenate(pooled_ll), np.concatenate(base_ll))}

    # 5. draw calibration
    hit = (df.y == 1).to_numpy()
    out["draw_calibration"] = {m: metrics.calibration_table(df[f"{m}_d"].to_numpy(), hit).to_dict("records") for m in MODELS}
    out["draw_rate"] = float(hit.mean())
    out["mean_draw_prob"] = {m: float(df[f"{m}_d"].mean()) for m in MODELS}

    # odds source note
    out["odds_sources"] = df.odds_src.value_counts().to_dict()

    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ["overall", "diff_bayes_minus_mle", "diff_bayes_minus_mkt",
                                          "diff_mle_minus_mkt", "subsets", "pool", "mean_draw_prob", "draw_rate"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
