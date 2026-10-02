"""Scoring rules and market helpers for home / draw / away probabilities.

Probability arrays are shaped (n, 3) in the order home, draw, away.
Outcomes are integer codes: 0 home win, 1 draw, 2 away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def outcome(hg, ag) -> np.ndarray:
    hg, ag = np.asarray(hg), np.asarray(ag)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def demargin(odds: np.ndarray) -> np.ndarray:
    """Proportional margin removal: implied probabilities rescaled to sum to 1."""
    implied = 1.0 / np.asarray(odds, dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p), 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((np.asarray(p) - onehot) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score: respects the ordering home < draw < away."""
    cp = np.cumsum(np.asarray(p), axis=1)[:, :2]
    co = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2))


def score_all(p: np.ndarray, y: np.ndarray) -> dict:
    return {"log_loss": log_loss(p, y), "rps": rps(p, y), "brier": brier(p, y), "n": int(len(y))}


def calibration_table(p_event: np.ndarray, hit: np.ndarray, bins=(0, .15, .2, .25, .3, .35, 1.0)) -> pd.DataFrame:
    """Mean forecast vs observed frequency in probability bins."""
    p_event, hit = np.asarray(p_event), np.asarray(hit).astype(float)
    idx = np.digitize(p_event, bins[1:-1])
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}", "n": int(m.sum()),
                         "forecast": float(p_event[m].mean()), "observed": float(hit[m].mean())})
    return pd.DataFrame(rows)


def bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) per match with a 95% bootstrap interval. Negative means a is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def per_match_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p), 1e-12, 1.0)
    return -np.log(p[np.arange(len(y)), y])
