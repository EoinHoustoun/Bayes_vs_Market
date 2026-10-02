"""Premier League results and closing odds from football-data.co.uk.

Every file is checked for Div == "E0" before use. In August 2026 a redirect
served National League matches (EC) under the Premier League URL, and nothing
downstream noticed, so the guard is not optional.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"

# Season label -> football-data season code
SEASONS = {
    "2016-17": "1617", "2017-18": "1718", "2018-19": "1819", "2019-20": "1920",
    "2020-21": "2021", "2021-22": "2122", "2022-23": "2223", "2023-24": "2324",
    "2024-25": "2425", "2025-26": "2526",
}

# Closing odds, sharpest first. Pinnacle closing where present, else market average closing.
CLOSING = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]


class WrongDivisionError(ValueError):
    """Raised when a file claims to be the Premier League but is not."""


def check_division(df: pd.DataFrame, expected: str = "E0") -> None:
    if "Div" not in df.columns:
        raise WrongDivisionError("No Div column: cannot verify this is the Premier League")
    divs = set(df["Div"].dropna().unique())
    if divs != {expected}:
        raise WrongDivisionError(f"Expected only {expected}, found {sorted(divs)}")
    if len(df) > 380:
        raise WrongDivisionError(f"{len(df)} matches is more than a Premier League season")


def _download(code: str) -> pd.DataFrame:
    r = requests.get(URL.format(code=code), timeout=30, allow_redirects=False)
    if r.status_code != 200:
        raise RuntimeError(f"football-data returned {r.status_code} for {code} (redirects are refused)")
    body = r.content[3:] if r.content.startswith(b"\xef\xbb\xbf") else r.content
    df = pd.read_csv(io.StringIO(body.decode("latin-1")))
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def load_season(label: str, refresh: bool = False) -> pd.DataFrame:
    """One season as a tidy frame: date, home, away, hg, ag, closing odds, season."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"E0_{SEASONS[label]}.csv"
    if refresh or not path.exists():
        raw = _download(SEASONS[label])
        check_division(raw)
        raw.to_csv(path, index=False)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    check_division(raw)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
    })
    out["oh"] = out["od"] = out["oa"] = float("nan")
    out["odds_src"] = ""
    for h, d, a in CLOSING:
        if {h, d, a} <= set(raw.columns):
            miss = out["oh"].isna()
            out.loc[miss, "oh"] = raw.loc[miss, h].values
            out.loc[miss, "od"] = raw.loc[miss, d].values
            out.loc[miss, "oa"] = raw.loc[miss, a].values
            out.loc[miss & raw[h].notna().values, "odds_src"] = h[:-1]
    out["season"] = label
    return out.sort_values("date").reset_index(drop=True)


def load_all(labels: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    labels = labels or list(SEASONS)
    df = pd.concat([load_season(s, refresh) for s in labels], ignore_index=True)
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    return add_promoted_flags(df)


def add_promoted_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag each side that was not in the Premier League the previous season.

    The first season in the data has no predecessor, so its flags are NaN and
    it should only ever be used as training data.
    """
    labels = sorted(df["season"].unique())
    teams = {s: set(df.loc[df.season == s, "home"]) for s in labels}
    df = df.copy()
    df["home_promoted"] = float("nan")
    df["away_promoted"] = float("nan")
    for prev, cur in zip(labels, labels[1:]):
        m = df.season == cur
        df.loc[m, "home_promoted"] = (~df.loc[m, "home"].isin(teams[prev])).astype(float)
        df.loc[m, "away_promoted"] = (~df.loc[m, "away"].isin(teams[prev])).astype(float)
    return df
