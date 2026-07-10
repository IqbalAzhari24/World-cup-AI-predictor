"""Predict a single match. Usage: python -m src.predict "France" "Morocco" [--home]"""
import argparse

import pandas as pd

from .elo import match_features
from .model import CLASSES, FEATURES, load_artifacts


def predict_match(home_team: str, away_team: str, neutral: bool = True):
    model, states = load_artifacts()
    for t in (home_team, away_team):
        if t not in states:
            raise KeyError(f"Unknown team: {t!r}")
    row = match_features(states[home_team], states[away_team], neutral)
    proba = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    return dict(zip(CLASSES, proba))


def main():
    p = argparse.ArgumentParser(description="Predict an international match")
    p.add_argument("home_team")
    p.add_argument("away_team")
    p.add_argument("--home", action="store_true",
                   help="first team plays at home (default: neutral venue, as in a World Cup)")
    args = p.parse_args()

    probs = predict_match(args.home_team, args.away_team, neutral=not args.home)
    venue = "at home" if args.home else "neutral venue"
    print(f"\n{args.home_team} vs {args.away_team}  ({venue})")
    print(f"  {args.home_team} win : {probs['home_win']:6.1%}")
    print(f"  Draw{'':<11}: {probs['draw']:6.1%}")
    print(f"  {args.away_team} win : {probs['away_win']:6.1%}")

    from .scoreline import top_scorelines
    try:
        (xg_h, xg_a), lines = top_scorelines(args.home_team, args.away_team, neutral=not args.home)
    except FileNotFoundError:
        return  # older artifact without goal models; W/D/L already printed
    print(f"\nExpected goals: {args.home_team} {xg_h:.2f} - {xg_a:.2f} {args.away_team}")
    print("Most likely scorelines:")
    for (gh, ga), p in lines:
        print(f"  {gh}-{ga}  {p:6.1%}")


if __name__ == "__main__":
    main()
