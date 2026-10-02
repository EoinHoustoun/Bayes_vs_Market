"""Walk-forward backtest: refit every 28 days, forecast the next 28 days.

For each test season, each block is forecast using only matches played before
the block starts (previous 3 years, exponentially down-weighted). Writes
results/predictions.csv and results/fit_log.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from numpyro.diagnostics import summary  # noqa: E402

from bf import data, metrics, models  # noqa: E402

TEST_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
XI = 0.0019          # per-day decay, half-life about a year (Dixon and Coles)
WINDOW_DAYS = 1095
BLOCK_DAYS = 28
OUT = Path(__file__).resolve().parent.parent / "results"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"   # "base" or "norm" (weights averaging 1)
SUFFIX = "" if VARIANT == "base" else f"_{VARIANT}"


def main():
    OUT.mkdir(exist_ok=True)
    df = data.load_all()
    teams = sorted(set(df.home) | set(df.away))
    idx = {t: i for i, t in enumerate(teams)}
    preds, fits = [], []
    t0 = time.time()
    for season in TEST_SEASONS:
        sea = df[df.season == season]
        start, end = sea.date.min(), sea.date.max()
        edges = list(pd.date_range(start, end + pd.Timedelta(days=1), freq=f"{BLOCK_DAYS}D")) + [end + pd.Timedelta(days=1)]
        for b0, b1 in zip(edges, edges[1:]):
            block = sea[(sea.date >= b0) & (sea.date < b1)]
            if block.empty:
                continue
            train = df[(df.date < b0) & (df.date >= b0 - pd.Timedelta(days=WINDOW_DAYS))]
            d = models.pack(train, idx, b0, XI, normalise=(VARIANT == "norm"))
            hi, ai = block.home.map(idx).to_numpy(), block.away.map(idx).to_numpy()
            hp, ap = block.home_promoted.to_numpy(), block.away_promoted.to_numpy()

            mle = models.fit_mle(d, len(teams))
            p_mle = models.predict(mle, hi, ai, hp, ap)
            samples, mcmc = models.fit_bayes(d, len(teams), seed=len(fits))
            p_bayes = models.predict(samples, hi, ai, hp, ap)

            sm = summary(mcmc.get_samples(group_by_chain=True))
            fits.append({
                "season": season, "block_start": b0.date(), "n_train": len(train), "n_test": len(block),
                "max_rhat": float(max(np.nanmax(sm[k]["r_hat"]) for k in ["att_raw", "dfn_raw", "mu", "home", "rho"])),
                "min_ess": float(min(np.nanmin(sm[k]["n_eff"]) for k in ["att_raw", "dfn_raw", "mu", "home", "rho"])),
                "divergences": int(mcmc.get_extra_fields()["diverging"].sum()),
                **{f"{k}_mean": float(samples[k].mean()) for k in ["home", "rho", "prom_att", "prom_dfn", "sigma_att", "sigma_dfn"]},
                **{f"{k}_lo": float(np.quantile(samples[k], .05)) for k in ["home", "prom_att", "prom_dfn"]},
                **{f"{k}_hi": float(np.quantile(samples[k], .95)) for k in ["home", "prom_att", "prom_dfn"]},
                "mle_home": float(mle["home"][0]), "mle_prom_att": float(mle["prom_att"][0]),
                "mle_prom_dfn": float(mle["prom_dfn"][0]), "mle_rho": float(mle["rho"][0]),
            })
            p_mkt = metrics.demargin(block[["oh", "od", "oa"]].to_numpy())
            out = block[["season", "date", "home", "away", "hg", "ag", "oh", "od", "oa", "odds_src",
                         "home_promoted", "away_promoted"]].copy()
            for name, p in [("mkt", p_mkt), ("mle", p_mle), ("bayes", p_bayes)]:
                out[[f"{name}_h", f"{name}_d", f"{name}_a"]] = p
            preds.append(out)
            pd.concat(preds).to_csv(OUT / f"predictions{SUFFIX}.csv", index=False)
            pd.DataFrame(fits).to_csv(OUT / f"fit_log{SUFFIX}.csv", index=False)
            print(f"[{time.time() - t0:6.0f}s] {season} {b0.date()} n_test={len(block):3d} "
                  f"rhat={fits[-1]['max_rhat']:.3f} div={fits[-1]['divergences']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
