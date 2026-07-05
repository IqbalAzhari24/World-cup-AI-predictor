"""Full-tournament Monte Carlo simulation: group stage + knockout.

Usage: python -m src.tournament [-n 10000] [--year 2026]

Groups are derived from the dataset itself: group-stage matches are the
edges that never grow a connected component beyond 4 teams, so the actual
draw falls out of the fixture list with no hardcoding.

Group games sample a scoreline (not just an outcome) so points, goal
difference and goals-for drive the standings like the real tiebreakers.
Top 2 per group advance, plus the 8 best third-placed teams when the
edition has 12 groups (2026 format). Qualifiers are seeded into a standard
knockout bracket by group-stage performance.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .data import load_matches
from .elo import match_features
from .model import FEATURES, load_artifacts


def extract_groups(year: int = 2026) -> list[list[str]]:
    """Recover the World Cup groups for `year` from the fixture graph."""
    df = pd.read_csv(load_matches.__globals__["DATA_PATH"], parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == year)].sort_values("date")
    if wc.empty:
        raise ValueError(f"No FIFA World Cup {year} matches in the dataset")

    # Union-find, refusing any merge that would exceed group size 4.
    parent: dict[str, str] = {}
    size: dict[str, int] = {}

    def find(x):
        parent.setdefault(x, x)
        size.setdefault(x, 1)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in wc.itertuples():
        a, b = find(m.home_team), find(m.away_team)
        if a != b and size[a] + size[b] <= 4:
            parent[a] = b
            size[b] += size[a]

    groups = defaultdict(list)
    for team in set(wc.home_team) | set(wc.away_team):
        groups[find(team)].append(team)
    result = sorted((sorted(g) for g in groups.values() if len(g) == 4), key=lambda g: g[0])
    if not result:
        raise ValueError(f"Could not recover groups of 4 for {year}")
    return result


def pairwise_probs(model, states, teams: list[str]) -> dict:
    """(a, b) -> (p_win, p_draw, p_loss) on neutral ground, for all ordered pairs."""
    pairs = [(a, b) for a in teams for b in teams if a != b]
    rows = [match_features(states[a], states[b], neutral=True) for a, b in pairs]
    proba = model.predict_proba(pd.DataFrame(rows)[FEATURES])
    return dict(zip(pairs, proba))


def sample_score(rng, p_win: float, p_draw: float):
    """Sample a plausible (home, away) scoreline consistent with a drawn outcome."""
    u = rng.random()
    if u < p_draw:
        g = rng.poisson(1.1)
        return g, g
    margin = 1 + rng.poisson(0.75)
    loser_goals = rng.poisson(0.9)
    if u < p_draw + p_win:
        return loser_goals + margin, loser_goals
    return loser_goals, loser_goals + margin


def knockout_positions(n: int) -> list[int]:
    """Standard bracket seeding: seed 0 can only meet seed 1 in the final."""
    order = [0]
    while len(order) < n:
        m = len(order) * 2
        order = [s for x in order for s in (x, m - 1 - x)]
    return order


def simulate_once(groups, probs, rng):
    """One full tournament. Returns (champion, runner_up)."""
    qualifiers = []  # (group_rank, points, gd, gf, team)
    thirds = []
    for group in groups:
        table = {t: [0, 0, 0] for t in group}  # points, gd, gf
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                p_win, p_draw, _ = probs[(a, b)]
                ga, gb = sample_score(rng, p_win, p_draw)
                table[a][0] += 3 if ga > gb else 1 if ga == gb else 0
                table[b][0] += 3 if gb > ga else 1 if ga == gb else 0
                table[a][1] += ga - gb
                table[b][1] += gb - ga
                table[a][2] += ga
                table[b][2] += gb
        ranked = sorted(group, key=lambda t: (*table[t], rng.random()), reverse=True)
        qualifiers += [(0, *table[ranked[0]], ranked[0]), (1, *table[ranked[1]], ranked[1])]
        thirds.append((2, *table[ranked[2]], ranked[2]))

    if len(groups) == 12:  # 2026 format: 8 best third-placed teams join
        thirds.sort(key=lambda q: q[1:4], reverse=True)
        qualifiers += thirds[:8]

    # Seed by group-stage performance into a standard bracket
    qualifiers.sort(key=lambda q: (q[0], -q[1], -q[2], -q[3]))
    seeds = [q[4] for q in qualifiers]
    alive = [seeds[i] for i in knockout_positions(len(seeds))]

    while len(alive) > 2:
        alive = [_knockout_winner(a, b, probs, rng) for a, b in zip(alive[::2], alive[1::2])]
    champion = _knockout_winner(alive[0], alive[1], probs, rng)
    return champion, alive[0] if champion == alive[1] else alive[1]


def _knockout_winner(a, b, probs, rng):
    p_win, p_draw, p_loss = probs[(a, b)]
    p_a = p_win + p_draw * (p_win / (p_win + p_loss))  # draws go to extra time / pens
    return a if rng.random() < p_a else b


def simulate_tournament(year: int = 2026, n_sims: int = 10000, seed: int = 42, artifacts=None):
    model, states = artifacts if artifacts is not None else load_artifacts()
    groups = extract_groups(year)
    teams = [t for g in groups for t in g]
    missing = [t for t in teams if t not in states]
    if missing:
        raise KeyError(f"Teams missing from ratings: {missing}")

    probs = pairwise_probs(model, states, teams)
    rng = np.random.default_rng(seed)
    champions, finalists = Counter(), Counter()
    for _ in range(n_sims):
        champion, runner_up = simulate_once(groups, probs, rng)
        champions[champion] += 1
        finalists[champion] += 1
        finalists[runner_up] += 1
    return champions, finalists


def main():
    ap = argparse.ArgumentParser(description="Simulate a full World Cup from the group stage")
    ap.add_argument("-n", "--sims", type=int, default=10000)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    groups = extract_groups(args.year)
    print(f"\nWorld Cup {args.year}: {len(groups)} groups")
    for i, g in enumerate(groups):
        print(f"  Group {chr(65 + i)}: {', '.join(g)}")

    champions, finalists = simulate_tournament(args.year, args.sims)
    print(f"\n{'Team':<16} {'Champion':>9} {'Finalist':>9}   ({args.sims} simulations)")
    for team, wins in champions.most_common(args.top):
        print(f"{team:<16} {wins / args.sims:>8.1%} {finalists[team] / args.sims:>8.1%}")


if __name__ == "__main__":
    main()
