"""Exact-scoreline probabilities from the Poisson expected-goals models."""
from math import exp, factorial

import numpy as np
import pandas as pd

from .elo import match_features
from .model import FEATURES, load_artifacts, load_goal_models

MAX_GOALS = 8  # per side; P(9+) is negligible for internationals


def poisson_pmf(lam: float, k: int) -> float:
    return exp(-lam) * lam**k / factorial(k)


def expected_goals(home_team: str, away_team: str, neutral: bool = True):
    """(xg_home, xg_away) for the fixture."""
    _, states = load_artifacts()
    xg_home, xg_away = load_goal_models()
    row = pd.DataFrame([match_features(states[home_team], states[away_team], neutral)])[FEATURES]
    return float(xg_home.predict(row)[0]), float(xg_away.predict(row)[0])


def scoreline_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    """P[i, j] = probability the score is home i, away j (independent Poissons)."""
    ph = np.array([poisson_pmf(lam_home, k) for k in range(MAX_GOALS + 1)])
    pa = np.array([poisson_pmf(lam_away, k) for k in range(MAX_GOALS + 1)])
    m = np.outer(ph, pa)
    return m / m.sum()  # renormalize the truncated tail


def top_scorelines(home_team: str, away_team: str, neutral: bool = True, n: int = 5):
    """Most likely scorelines: [((home_goals, away_goals), probability), ...]."""
    lam_h, lam_a = expected_goals(home_team, away_team, neutral)
    m = scoreline_matrix(lam_h, lam_a)
    flat = [((i, j), m[i, j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return (lam_h, lam_a), flat[:n]
