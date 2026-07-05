"""Monte Carlo knockout-bracket simulation.

Usage: python -m src.simulate "Argentina" "Egypt" "Switzerland" "Colombia" ... -n 10000

Teams are given in bracket order: (1 vs 2), (3 vs 4), winners meet, and so on.
Team count must be a power of two. Draws are resolved by re-splitting the draw
probability between the two teams (a stand-in for extra time / penalties).
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .elo import match_features
from .model import FEATURES, load_artifacts


def win_probability(model, states, team_a: str, team_b: str) -> float:
    """P(team_a beats team_b) on neutral ground, draw mass split proportionally."""
    row = match_features(states[team_a], states[team_b], neutral=True)
    p_win, p_draw, p_loss = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    return p_win + p_draw * (p_win / (p_win + p_loss))


def simulate_knockout(teams: list[str], n_sims: int = 10000, seed: int = 42, artifacts=None) -> Counter:
    model, states = artifacts if artifacts is not None else load_artifacts()
    for t in teams:
        if t not in states:
            raise KeyError(f"Unknown team: {t!r}")

    # Precompute pairwise win probabilities for every possible matchup
    p = {
        (a, b): win_probability(model, states, a, b)
        for a in teams for b in teams if a != b
    }

    rng = np.random.default_rng(seed)
    champions = Counter()
    for _ in range(n_sims):
        alive = list(teams)
        while len(alive) > 1:
            alive = [
                a if rng.random() < p[(a, b)] else b
                for a, b in zip(alive[::2], alive[1::2])
            ]
        champions[alive[0]] += 1
    return champions


def main():
    ap = argparse.ArgumentParser(description="Simulate a knockout bracket")
    ap.add_argument("teams", nargs="+", help="teams in bracket order (power of two)")
    ap.add_argument("-n", "--sims", type=int, default=10000)
    args = ap.parse_args()

    if len(args.teams) & (len(args.teams) - 1):
        ap.error("number of teams must be a power of two")

    champions = simulate_knockout(args.teams, args.sims)
    print(f"\nChampion probabilities ({args.sims} simulations):")
    for team, wins in champions.most_common():
        print(f"  {team:<20} {wins / args.sims:6.1%}")


if __name__ == "__main__":
    main()
