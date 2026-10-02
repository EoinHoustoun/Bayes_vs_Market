"""Two goal models with the same structure, fitted two ways.

Both model home and away goals as Poisson with team attack and defence terms,
home advantage, a shared promoted-side effect and the Dixon-Coles correction
for low scores. Matches are down-weighted exponentially with age.

* ``fit_mle``  : maximum likelihood (the classical Dixon-Coles approach).
                 A team with no data sits at league average.
* ``fit_bayes``: hierarchical Bayesian version sampled with NUTS. Team
                 strengths share a learned spread (partial pooling), and a
                 team with no data is drawn from that spread plus the
                 promoted-side effect, so its uncertainty is carried into
                 every forecast.

Arrays are padded to fixed shapes so JAX compiles the sampler once.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(2)

MAX_GOALS = 10
N_PAD = 1600  # > 4 seasons of matches; padding rows carry zero weight


# ---------------------------------------------------------------- shared maths
def _poisson_logpmf(k, lam):
    return k * jnp.log(lam) - lam - gammaln(k + 1.0)


def _log_tau(hg, ag, lam, mu, rho):
    """Dixon-Coles adjustment for the four low-scoring results."""
    tau = jnp.where((hg == 0) & (ag == 0), 1.0 - lam * mu * rho,
          jnp.where((hg == 0) & (ag == 1), 1.0 + lam * rho,
          jnp.where((hg == 1) & (ag == 0), 1.0 + mu * rho,
          jnp.where((hg == 1) & (ag == 1), 1.0 - rho, 1.0))))
    return jnp.log(jnp.clip(tau, 1e-10, None))


def _rates(p, hi, ai, hp, ap):
    lam = jnp.exp(p["mu"] + p["home"] + p["att"][hi] + p["dfn"][ai] + p["prom_att"] * hp + p["prom_dfn"] * ap)
    mu = jnp.exp(p["mu"] + p["att"][ai] + p["dfn"][hi] + p["prom_att"] * ap + p["prom_dfn"] * hp)
    return lam, mu


def _loglik(p, d):
    lam, mu = _rates(p, d["hi"], d["ai"], d["hp"], d["ap"])
    ll = (_poisson_logpmf(d["hg"], lam) + _poisson_logpmf(d["ag"], mu)
          + _log_tau(d["hg"], d["ag"], lam, mu, p["rho"]))
    return jnp.sum(d["w"] * ll)


def _centre(x, mask):
    return x - jnp.sum(x * mask) / jnp.sum(mask)


# ---------------------------------------------------------------- packing
def pack(train, team_index, ref_date, xi, normalise=False):
    """Fixed-shape arrays for a training window. Weight = exp(-xi * age in days).

    ``normalise`` rescales weights to average 1, so the total evidence equals the
    number of matches while keeping the recency profile. The MLE is unaffected;
    the Bayesian posterior is not, because shrinkage depends on how much data it sees.
    """
    n = len(train)
    if n > N_PAD:
        raise ValueError(f"{n} training matches exceeds N_PAD={N_PAD}")
    age = (ref_date - train["date"]).dt.days.to_numpy(float)
    w = np.zeros(N_PAD); w[:n] = np.exp(-xi * age)
    if normalise and n:
        w[:n] *= n / w[:n].sum()
    z = lambda col, f=float: np.concatenate([train[col].to_numpy(f), np.zeros(N_PAD - n, f)])
    hi = np.zeros(N_PAD, int); hi[:n] = train["home"].map(team_index).to_numpy()
    ai = np.zeros(N_PAD, int); ai[:n] = train["away"].map(team_index).to_numpy()
    mask = np.zeros(len(team_index))
    mask[np.unique(np.concatenate([hi[:n], ai[:n]]))] = 1.0
    return {"hi": jnp.asarray(hi), "ai": jnp.asarray(ai), "hg": jnp.asarray(z("hg")), "ag": jnp.asarray(z("ag")),
            "hp": jnp.asarray(np.nan_to_num(z("home_promoted"))), "ap": jnp.asarray(np.nan_to_num(z("away_promoted"))),
            "w": jnp.asarray(w), "mask": jnp.asarray(mask)}


# ---------------------------------------------------------------- MLE
def fit_mle(d, n_teams, ridge=1e-4):
    """Weighted maximum likelihood. Returns a dict of point estimates (one 'sample')."""
    def unpack(v):
        att = _centre(v[:n_teams], d["mask"])
        dfn = _centre(v[n_teams:2 * n_teams], d["mask"])
        mu, home, pa, pd_, rho = v[2 * n_teams:]
        return {"att": att, "dfn": dfn, "mu": mu, "home": home, "prom_att": pa, "prom_dfn": pd_,
                "rho": 0.2 * jnp.tanh(rho)}

    def nll(v):
        return -_loglik(unpack(v), d) / jnp.sum(d["w"]) + ridge * jnp.sum(v[:2 * n_teams] ** 2)

    f = jax.jit(jax.value_and_grad(nll))
    x0 = np.zeros(2 * n_teams + 5); x0[2 * n_teams + 1] = 0.25
    res = minimize(lambda v: tuple(np.asarray(t, float) for t in f(jnp.asarray(v))), x0, jac=True, method="L-BFGS-B")
    p = unpack(jnp.asarray(res.x))
    return {k: np.asarray(v)[None, ...] for k, v in p.items()}


# ---------------------------------------------------------------- Bayesian
def bayes_model(d, n_teams):
    mu = numpyro.sample("mu", dist.Normal(0.1, 0.5))
    home = numpyro.sample("home", dist.Normal(0.25, 0.25))
    s_att = numpyro.sample("sigma_att", dist.HalfNormal(0.5))
    s_dfn = numpyro.sample("sigma_dfn", dist.HalfNormal(0.5))
    with numpyro.plate("teams", n_teams):
        att_raw = numpyro.sample("att_raw", dist.Normal(0, 1))
        dfn_raw = numpyro.sample("dfn_raw", dist.Normal(0, 1))
    att = numpyro.deterministic("att", _centre(att_raw * s_att, d["mask"]))
    dfn = numpyro.deterministic("dfn", _centre(dfn_raw * s_dfn, d["mask"]))
    prom_att = numpyro.sample("prom_att", dist.Normal(0, 0.5))
    prom_dfn = numpyro.sample("prom_dfn", dist.Normal(0, 0.5))
    rho = numpyro.sample("rho", dist.Uniform(-0.2, 0.2))
    p = {"att": att, "dfn": dfn, "mu": mu, "home": home, "prom_att": prom_att, "prom_dfn": prom_dfn, "rho": rho}
    numpyro.factor("weighted_loglik", _loglik(p, d))


def fit_bayes(d, n_teams, draws=600, warmup=600, chains=2, seed=0):
    mcmc = MCMC(NUTS(bayes_model, target_accept_prob=0.85), num_warmup=warmup, num_samples=draws,
                num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), d=d, n_teams=n_teams)
    s = mcmc.get_samples()
    keep = ["att", "dfn", "mu", "home", "prom_att", "prom_dfn", "rho", "sigma_att", "sigma_dfn"]
    return {k: np.asarray(s[k]) for k in keep}, mcmc


# ---------------------------------------------------------------- prediction
def predict(samples, hi, ai, hp, ap, chunk=200):
    """Home/draw/away probabilities averaged over posterior samples (posterior predictive).

    ``samples`` values have a leading sample axis; MLE passes a single 'sample'.
    """
    hi, ai = np.asarray(hi), np.asarray(ai)
    hp, ap = np.nan_to_num(np.asarray(hp, float)), np.nan_to_num(np.asarray(ap, float))
    g = np.arange(MAX_GOALS + 1)
    S = len(samples["mu"])
    acc = np.zeros((len(hi), 3))
    from scipy.stats import poisson
    for s0 in range(0, S, chunk):
        sl = slice(s0, s0 + chunk)
        att, dfn = samples["att"][sl], samples["dfn"][sl]
        base = samples["mu"][sl][:, None]
        lam = np.exp(base + samples["home"][sl][:, None] + att[:, hi] + dfn[:, ai]
                     + samples["prom_att"][sl][:, None] * hp + samples["prom_dfn"][sl][:, None] * ap)
        mu = np.exp(base + att[:, ai] + dfn[:, hi]
                    + samples["prom_att"][sl][:, None] * ap + samples["prom_dfn"][sl][:, None] * hp)
        ph = poisson.pmf(g[None, None, :], lam[..., None])           # (s, m, G)
        pa = poisson.pmf(g[None, None, :], mu[..., None])
        grid = ph[..., :, None] * pa[..., None, :]                    # (s, m, G, G)
        rho = samples["rho"][sl][:, None]
        grid[..., 0, 0] *= 1 - lam * mu * rho
        grid[..., 0, 1] *= 1 + lam * rho
        grid[..., 1, 0] *= 1 + mu * rho
        grid[..., 1, 1] *= 1 - rho
        grid /= grid.sum(axis=(-1, -2), keepdims=True)
        home = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1)
        draw = np.eye(MAX_GOALS + 1)
        acc += np.stack([(grid * home).sum((-1, -2)), (grid * draw).sum((-1, -2)),
                         (grid * home.T).sum((-1, -2))], axis=-1).sum(axis=0)
    return acc / S
