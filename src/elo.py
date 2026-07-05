"""Elo ratings and rolling-form features computed chronologically over all matches.

Every feature attached to a match uses only information available *before* that
match was played, so the training data has no leakage.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 60.0  # Elo bonus for a non-neutral home team
FORM_WINDOW = 5  # matches used for recent-form features


def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return 60.0
    if "friendly" in t:
        return 20.0
    return 40.0  # qualifiers, continental cups, nations leagues


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    return np.log2(abs(goal_diff) + 1.0) or 1.0


class TeamState:
    """Tracks one team's rating and recent results."""

    def __init__(self):
        self.rating = BASE_RATING
        self.points = deque(maxlen=FORM_WINDOW)  # 3 win / 1 draw / 0 loss
        self.scored = deque(maxlen=FORM_WINDOW)
        self.conceded = deque(maxlen=FORM_WINDOW)

    def form(self) -> float:
        return float(np.mean(self.points)) if self.points else 1.0

    def avg_scored(self) -> float:
        return float(np.mean(self.scored)) if self.scored else 1.0

    def avg_conceded(self) -> float:
        return float(np.mean(self.conceded)) if self.conceded else 1.0


def compute_features(df: pd.DataFrame):
    """Walk matches in date order, emitting pre-match features and updating state.

    Returns (features_df, states) where states maps team name -> TeamState with
    ratings as of the end of the data.
    """
    states: dict[str, TeamState] = defaultdict(TeamState)
    rows = []

    for m in df.itertuples(index=False):
        home, away = states[m.home_team], states[m.away_team]
        home_elo = home.rating + (0.0 if m.neutral else HOME_ADVANTAGE)

        rows.append(
            {
                "elo_home": home.rating,
                "elo_away": away.rating,
                "elo_diff": home_elo - away.rating,
                "neutral": float(m.neutral),
                "form_diff": home.form() - away.form(),
                "attack_diff": home.avg_scored() - away.avg_scored(),
                "defense_diff": home.avg_conceded() - away.avg_conceded(),
            }
        )

        # Update ratings after the match
        result = 1.0 if m.home_score > m.away_score else 0.0 if m.home_score < m.away_score else 0.5
        k = k_factor(m.tournament) * margin_multiplier(m.home_score - m.away_score)
        delta = k * (result - expected_score(home_elo, away.rating))
        home.rating += delta
        away.rating -= delta

        home.points.append(3 if result == 1.0 else 1 if result == 0.5 else 0)
        away.points.append(3 if result == 0.0 else 1 if result == 0.5 else 0)
        home.scored.append(m.home_score)
        home.conceded.append(m.away_score)
        away.scored.append(m.away_score)
        away.conceded.append(m.home_score)

    return pd.DataFrame(rows), dict(states)


def match_features(home: TeamState, away: TeamState, neutral: bool) -> dict:
    """Feature row for a hypothetical future match, from current team states."""
    home_elo = home.rating + (0.0 if neutral else HOME_ADVANTAGE)
    return {
        "elo_home": home.rating,
        "elo_away": away.rating,
        "elo_diff": home_elo - away.rating,
        "neutral": float(neutral),
        "form_diff": home.form() - away.form(),
        "attack_diff": home.avg_scored() - away.avg_scored(),
        "defense_diff": home.avg_conceded() - away.avg_conceded(),
    }
