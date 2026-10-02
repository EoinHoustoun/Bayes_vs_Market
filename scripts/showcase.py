"""Fit once on everything up to the end of 2025-26 and export posterior summaries.

Writes results/team_strengths.csv (attack and defence with 90% intervals for the
2025-26 teams) and results/showcase.json (global parameters).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bf import data, models  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    df = data.load_all()
    teams = sorted(set(df.home) | set(df.away))
    idx = {t: i for i, t in enumerate(teams)}
    ref = pd.Timestamp("2026-06-01")
    train = df[df.date >= ref - pd.Timedelta(days=1095)]
    s, mcmc = models.fit_bayes(models.pack(train, idx, ref, 0.0019, normalise=True), len(teams), draws=1000, warmup=800, seed=42)
    mle = models.fit_mle(models.pack(train, idx, ref, 0.0019), len(teams))

    current = sorted(set(df.loc[df.season == "2025-26", "home"]))
    sea = df[df.season == "2025-26"]
    promoted = set(sea.loc[sea.home_promoted == 1, "home"])
    rows = []
    for t in current:
        i = idx[t]
        # effective 2025-26 level: promoted sides carry the shared promoted-season effect
        a = s["att"][:, i] + (s["prom_att"] if t in promoted else 0.0)
        dfn = s["dfn"][:, i] + (s["prom_dfn"] if t in promoted else 0.0)
        rows.append({"team": t,
                     "attack": float(a.mean()), "attack_lo": float(np.quantile(a, .05)), "attack_hi": float(np.quantile(a, .95)),
                     "defence": float(-dfn.mean()), "defence_lo": float(np.quantile(-dfn, .05)), "defence_hi": float(np.quantile(-dfn, .95)),
                     "mle_attack": float(mle["att"][0, i] + (mle["prom_att"][0] if t in promoted else 0.0)),
                     "mle_defence": float(-(mle["dfn"][0, i] + (mle["prom_dfn"][0] if t in promoted else 0.0))),
                     "matches_in_window": int(((train.home == t) | (train.away == t)).sum()), "promoted_2025_26": t in promoted})
    pd.DataFrame(rows).sort_values("attack", ascending=False).to_csv(RES / "team_strengths.csv", index=False)

    q = lambda x: {"mean": float(np.mean(x)), "lo": float(np.quantile(x, .05)), "hi": float(np.quantile(x, .95))}
    out = {k: q(s[k]) for k in ["home", "rho", "prom_att", "prom_dfn", "sigma_att", "sigma_dfn", "mu"]}
    out["promoted_goals_scored_multiplier"] = q(np.exp(s["prom_att"]))
    out["promoted_goals_conceded_multiplier"] = q(np.exp(s["prom_dfn"]))
    out["home_goals_multiplier"] = q(np.exp(s["home"]))
    out["divergences"] = int(mcmc.get_extra_fields()["diverging"].sum())
    (RES / "showcase.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
