"""Load the international match results dataset."""
from pathlib import Path
import urllib.request

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "results.csv"


def load_matches(refresh: bool = False) -> pd.DataFrame:
    """Return all played matches sorted by date, downloading the CSV if needed."""
    if refresh or not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures() -> pd.DataFrame:
    """Return scheduled matches (no score yet), e.g. upcoming World Cup games."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df[df["home_score"].isna()].sort_values("date").reset_index(drop=True)
