import numpy as np
import pandas as pd
import pytest

from bf import data, metrics, models


def test_division_guard_rejects_national_league():
    df = pd.DataFrame({"Div": ["EC"] * 5})
    with pytest.raises(data.WrongDivisionError):
        data.check_division(df)


def test_division_guard_rejects_mixed_and_oversized():
    with pytest.raises(data.WrongDivisionError):
        data.check_division(pd.DataFrame({"Div": ["E0", "E1"]}))
    with pytest.raises(data.WrongDivisionError):
        data.check_division(pd.DataFrame({"Div": ["E0"] * 381}))
    data.check_division(pd.DataFrame({"Div": ["E0"] * 380}))


def test_demargin_sums_to_one_and_keeps_order():
    p = metrics.demargin(np.array([[2.10, 3.40, 3.60]]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0, 0] > p[0, 1] > p[0, 2]


def test_rps_perfect_and_worst():
    y = np.array([0, 2])
    assert metrics.rps(np.array([[1, 0, 0], [0, 0, 1]]), y) == pytest.approx(0.0)
    assert metrics.rps(np.array([[0, 0, 1], [1, 0, 0]]), y) == pytest.approx(1.0)


def test_log_loss_uniform():
    y = np.array([0, 1, 2])
    assert metrics.log_loss(np.full((3, 3), 1 / 3), y) == pytest.approx(np.log(3))


def test_outcome_codes():
    assert list(metrics.outcome([2, 1, 0], [1, 1, 3])) == [0, 1, 2]


def test_promoted_flags():
    df = pd.DataFrame({
        "season": ["A", "A", "B", "B"], "date": pd.to_datetime(["2020-01-01"] * 4),
        "home": ["X", "Y", "X", "Z"], "away": ["Y", "X", "Z", "X"], "hg": 0, "ag": 0,
    })
    out = data.add_promoted_flags(df)
    b = out[out.season == "B"]
    assert list(b.home_promoted) == [0.0, 1.0]
    assert list(b.away_promoted) == [1.0, 0.0]
    assert out[out.season == "A"].home_promoted.isna().all()


def test_predict_probabilities_sum_to_one_and_home_advantage():
    S, T = 50, 4
    samples = {"att": np.zeros((S, T)), "dfn": np.zeros((S, T)), "mu": np.full(S, 0.2),
               "home": np.full(S, 0.3), "prom_att": np.zeros(S), "prom_dfn": np.zeros(S), "rho": np.full(S, -0.05)}
    p = models.predict(samples, [0, 1], [1, 0], [0, 0], [0, 0])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p[:, 0] > p[:, 2]).all()


def test_mle_recovers_stronger_team():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(400):
        h, a = rng.choice(4, 2, replace=False)
        strength = np.array([0.5, 0.0, 0.0, -0.5])
        rows.append({"home": f"T{h}", "away": f"T{a}", "hg": rng.poisson(np.exp(0.3 + strength[h])),
                     "ag": rng.poisson(np.exp(strength[a])), "date": pd.Timestamp("2021-01-01"),
                     "home_promoted": 0.0, "away_promoted": 0.0})
    train = pd.DataFrame(rows)
    idx = {f"T{i}": i for i in range(4)}
    d = models.pack(train, idx, pd.Timestamp("2021-01-02"), xi=0.0)
    p = models.fit_mle(d, 4)
    att = p["att"][0]
    assert att[0] > att[1] > att[3]
