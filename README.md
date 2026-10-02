# Bayes vs the Market

A hierarchical Bayesian goal model (MCMC), a classical Dixon-Coles model and a tabular foundation model, tested walk-forward on five Premier League seasons against the closing betting market.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/) [![NumPyro](https://img.shields.io/badge/NumPyro-NUTS-orange)](https://num.pyro.ai/) [![JAX](https://img.shields.io/badge/JAX-0.4-green)](https://github.com/google/jax) [![TabICL](https://img.shields.io/badge/TabICL-v2-teal)](https://github.com/soda-inria/tabicl)

**Case study with interactive charts:** [eoinhoustoun.github.io/case-studies/bayes-vs-market.html](https://eoinhoustoun.github.io/case-studies/bayes-vs-market.html)

## Overview
- **Problem:** does a Bayesian model, with partial pooling and full posterior uncertainty, forecast Premier League results better than maximum likelihood? Can a zero-tuning tabular foundation model beat both?
- **Approach:** the same Poisson goal model (team attack and defence, home advantage, Dixon-Coles low-score correction, a shared promoted-side effect, exponential time decay) fitted by maximum likelihood and by NUTS in NumPyro; TabICL v2 stacked on top. Refit every 28 days on the previous three years and scored on the next 28 days.
- **Result:** on 1,520 out-of-sample matches the closing market wins (log loss 0.961). The Bayesian model (0.983) and classical model (0.985) finish level; TabICL adds nothing beyond the market. The Bayesian model uniquely finds that promoted sides score about 30% fewer goals and concede about 41% more than their later level.

## Key Features
- Hierarchical Bayesian model sampled with NUTS: 108 fits, maximum R-hat 1.035, zero divergences
- Strict walk-forward validation: every forecast uses only matches played before it
- Paired bootstrap intervals for every comparison between forecasters
- Diagnosed and fixed over-shrinkage: time-decay weights cut the effective sample, which a Bayesian posterior (unlike maximum likelihood) is sensitive to; one pre-declared fix brought the model level with the classical fit
- Tabular foundation model benchmark (TabICL v2, BSD-3 licence) with and without the market as an input
- Data guard rejects any file that is not Premier League data

## Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Models | Hierarchical Poisson with Dixon-Coles correction, maximum likelihood, TabICL v2 |
| Inference | NumPyro (NUTS), JAX, SciPy (L-BFGS) |
| Libraries | pandas, NumPy, ArviZ, PyTorch (TabICL) |
| Visualisation | Hand-built SVG case study page |

## Results
Log loss on 1,520 matches, 2022-23 to 2025-26 (lower is better):

| Forecaster | Log loss | Gap to market (95% interval) |
|---|---|---|
| Closing market | 0.9605 | |
| TabICL, models + market | 0.9682 | +0.008 (+0.002 to +0.014) |
| Bayesian, fixed weights | 0.9833 | +0.023 (+0.013 to +0.032) |
| Classical (maximum likelihood) | 0.9850 | +0.025 (+0.015 to +0.034) |
| Bayesian, first run | 0.9897 | +0.029 (+0.018 to +0.040) |
| TabICL, models only | 0.9899 | +0.029 (+0.018 to +0.041) |

Posterior from the final fit (90% intervals): home sides score 1.19x (1.12 to 1.26) the goals; promoted sides score 0.70x (0.59 to 0.82) and concede 1.41x (1.21 to 1.63) compared with their later level.

## How to Run
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                 # 9 tests
.venv/bin/python scripts/backtest.py norm      # walk-forward backtest (about 45 minutes on a laptop CPU)
.venv/bin/python scripts/analyse.py
.venv/bin/python scripts/showcase.py

# TabICL needs its own environment (PyTorch 2.2 on Intel Macs)
uv venv --python 3.11 .venv-tabicl && uv pip install --python .venv-tabicl/bin/python tabicl "torch==2.2.2" "numpy<2" "pandas<2.3" scikit-learn
.venv-tabicl/bin/python scripts/tabicl_stack.py norm
```
Match data downloads from football-data.co.uk on first run.

## Project Structure
```
bf/data.py        download, division guard, promoted-side flags
bf/models.py      maximum likelihood and NumPyro hierarchical model, posterior predictive
bf/metrics.py     log loss, RPS, Brier, calibration, bootstrap
scripts/          backtest, analysis, showcase fit, TabICL stacking
results/          forecasts, fit diagnostics, summaries
writeup/          case study page
tests/            unit tests
```

---
*Part of Eoin Houstoun's Data Science Portfolio, [github.com/EoinHoustoun](https://github.com/EoinHoustoun)*
