"""Train the match-outcome model: features -> P(home win), P(draw), P(away win)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss

from .data import load_matches
from .elo import compute_features

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

FEATURES = ["elo_home", "elo_away", "elo_diff", "neutral", "form_diff", "attack_diff", "defense_diff"]
CLASSES = ["home_win", "draw", "away_win"]


def outcome(home_score: int, away_score: int) -> int:
    return 0 if home_score > away_score else 1 if home_score == away_score else 2


def train(min_year: int = 1990, test_years: int = 4, verbose: bool = True):
    """Train on matches since `min_year`, holding out the last `test_years` for evaluation."""
    matches = load_matches()
    feats, states = compute_features(matches)
    y = np.array([outcome(h, a) for h, a in zip(matches.home_score, matches.away_score)])

    mask = matches.date.dt.year >= min_year
    X, y, dates = feats[mask][FEATURES], y[mask.values], matches.date[mask]

    cutoff = dates.max() - pd.DateOffset(years=test_years)
    train_idx, test_idx = (dates <= cutoff).values, (dates > cutoff).values

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X[train_idx], y[train_idx])

    if verbose:
        proba = model.predict_proba(X[test_idx])
        print(f"Train: {train_idx.sum()} matches   Test (last {test_years}y): {test_idx.sum()} matches")
        print(f"Test accuracy: {accuracy_score(y[test_idx], proba.argmax(1)):.3f}")
        print(f"Test log loss: {log_loss(y[test_idx], proba):.3f}")
        baseline = np.bincount(y[train_idx], minlength=3) / train_idx.sum()
        print(f"(always-predict-prior baseline log loss: "
              f"{log_loss(y[test_idx], np.tile(baseline, (test_idx.sum(), 1))):.3f})")

    # Poisson goal models: expected goals for each side, for scoreline prediction
    goals_home = matches.home_score[mask].clip(upper=10)
    goals_away = matches.away_score[mask].clip(upper=10)
    xg_home = HistGradientBoostingRegressor(
        loss="poisson", max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    xg_away = HistGradientBoostingRegressor(
        loss="poisson", max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    xg_home.fit(X[train_idx], goals_home[train_idx])
    xg_away.fit(X[train_idx], goals_away[train_idx])
    if verbose:
        mae_h = np.abs(xg_home.predict(X[test_idx]) - goals_home[test_idx]).mean()
        mae_a = np.abs(xg_away.predict(X[test_idx]) - goals_away[test_idx]).mean()
        print(f"Expected-goals MAE (test): home {mae_h:.3f}, away {mae_a:.3f}")

    # Refit on everything before saving so predictions use all available history
    model.fit(X, y)
    xg_home.fit(X, goals_home)
    xg_away.fit(X, goals_away)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump({"model": model, "xg_home": xg_home, "xg_away": xg_away, "states": states}, f)
    if verbose:
        print(f"Saved model + ratings for {len(states)} teams -> {ARTIFACT_PATH}")
    return model, states


def load_artifacts():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python -m src.model")
    with open(ARTIFACT_PATH, "rb") as f:
        art = pickle.load(f)
    return art["model"], art["states"]


def load_goal_models():
    """Poisson expected-goals regressors (retrain if the artifact predates them)."""
    with open(ARTIFACT_PATH, "rb") as f:
        art = pickle.load(f)
    if "xg_home" not in art:
        raise FileNotFoundError("Model artifact has no goal models. Re-run: python -m src.model")
    return art["xg_home"], art["xg_away"]


if __name__ == "__main__":
    train()
