"""Player-level match view: first XI, goalscorers, man of the match.

Built on top of a football-data.org /matches/{id} response (see src/football_data.py).
football-data.org does not publish an official "man of the match" award, so
`man_of_the_match` derives one from goals/assists/cards — it's a heuristic,
not an official stat.
"""
import argparse

import pandas as pd

from src.football_data import COMPETITIONS, api_key_from_env, get_match, list_matches

GOAL_POINTS = 4
OWN_GOAL_POINTS = -2
ASSIST_POINTS = 2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
WINNING_SIDE_BONUS = 1


def match_summary(match: dict) -> str:
    home, away = match["homeTeam"]["name"], match["awayTeam"]["name"]
    full = match.get("score", {}).get("fullTime", {})
    h, a = full.get("home"), full.get("away")
    score = f"{h}-{a}" if h is not None and a is not None else "?-?"
    return f"{home} {score} {away} ({match['utcDate'][:10]})"


def first_xi(match: dict) -> dict[str, dict]:
    """Return {'home': {'formation': ..., 'lineup': DataFrame}, 'away': {...}}.

    `lineup` is empty if football-data.org has no lineup data for this match
    (common for lower tiers/older matches even on a paid plan).
    """
    out = {}
    for side in ("home", "away"):
        team = match[f"{side}Team"]
        players = team.get("lineup") or []
        df = pd.DataFrame(
            [
                {
                    "shirt": p.get("shirtNumber"),
                    "name": p.get("name"),
                    "position": p.get("position"),
                }
                for p in players
            ]
        )
        if not df.empty:
            df = df.sort_values("shirt", na_position="last").reset_index(drop=True)
        out[side] = {"team": team.get("name"), "formation": team.get("formation"), "lineup": df}
    return out


def goalscorers(match: dict) -> pd.DataFrame:
    goals = match.get("goals") or []
    rows = []
    for g in goals:
        scorer = g.get("scorer") or {}
        assist = g.get("assist") or {}
        rows.append(
            {
                "minute": g.get("minute"),
                "team": (g.get("team") or {}).get("name"),
                "scorer": scorer.get("name"),
                "assist": assist.get("name"),
                "type": g.get("type"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("minute").reset_index(drop=True)
    return df


def man_of_the_match(match: dict) -> dict | None:
    """Heuristic best-player pick from goals, assists and cards.

    Not an official award — football-data.org doesn't provide one. Returns
    None when the match has no scorable events at all (no goals or cards),
    e.g. a 0-0 with no bookings, or a match with no event data available.
    """
    full = match.get("score", {}).get("fullTime", {})
    home_goals, away_goals = full.get("home"), full.get("away")
    winning_team = None
    if home_goals is not None and away_goals is not None and home_goals != away_goals:
        winning_team = match["homeTeam"]["name"] if home_goals > away_goals else match["awayTeam"]["name"]

    scores: dict[str, dict] = {}

    def bump(player: dict, team_name: str, points: int):
        name = player.get("name")
        if not name:
            return
        entry = scores.setdefault(name, {"name": name, "team": team_name, "points": 0})
        entry["points"] += points

    for g in match.get("goals") or []:
        scorer, assist = g.get("scorer") or {}, g.get("assist") or {}
        team_name = (g.get("team") or {}).get("name")
        bump(scorer, team_name, OWN_GOAL_POINTS if g.get("type") == "OWN" else GOAL_POINTS)
        if assist.get("name"):
            bump(assist, team_name, ASSIST_POINTS)

    for b in match.get("bookings") or []:
        player = b.get("player") or {}
        team_name = (b.get("team") or {}).get("name")
        points = RED_CARD_POINTS if b.get("card") == "RED_CARD" else YELLOW_CARD_POINTS
        bump(player, team_name, points)

    if not scores:
        return None

    if winning_team:
        for entry in scores.values():
            if entry["team"] == winning_team:
                entry["points"] += WINNING_SIDE_BONUS

    return max(scores.values(), key=lambda e: e["points"])


def _print_match(match: dict):
    print(match_summary(match))

    xi = first_xi(match)
    for side in ("home", "away"):
        info = xi[side]
        formation = f" ({info['formation']})" if info["formation"] else ""
        print(f"\n{info['team']} — starting XI{formation}:")
        if info["lineup"].empty:
            print("  (no lineup data available for this match)")
        else:
            for _, row in info["lineup"].iterrows():
                shirt = f"{int(row['shirt']):>2} " if pd.notna(row["shirt"]) else "   "
                print(f"  {shirt}{row['name']} — {row['position']}")

    goals = goalscorers(match)
    print("\nGoals:")
    if goals.empty:
        print("  (none)")
    else:
        for _, row in goals.iterrows():
            assist = f" (assist: {row['assist']})" if pd.notna(row["assist"]) else ""
            print(f"  {row['minute']}' {row['scorer']} — {row['team']}{assist}")

    motm = man_of_the_match(match)
    print("\nMan of the match (heuristic, not an official award):")
    print(f"  {motm['name']} ({motm['team']}) — score {motm['points']}" if motm else "  (not enough match data)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("competition", nargs="?", default="WC", choices=COMPETITIONS,
                         help="Competition code, e.g. WC (World Cup), EC, CL, PL. Default: WC")
    parser.add_argument("--matchday", type=int, help="Only this matchday")
    parser.add_argument("--match-id", type=int, help="Skip the picker, show one match directly")
    parser.add_argument("--api-key", default=None, help="Defaults to FOOTBALL_DATA_API_KEY env var")
    args = parser.parse_args()

    api_key = args.api_key or api_key_from_env()

    if args.match_id:
        _print_match(get_match(args.match_id, api_key))
        return

    matches = list_matches(args.competition, api_key, matchday=args.matchday)
    if not matches:
        print(f"No finished matches found for {COMPETITIONS[args.competition]}.")
        return
    # Each match needs its own request for lineup/goal detail, and the free
    # tier is capped at 10 requests/minute — keep this small by default.
    for m in matches[:5]:
        _print_match(get_match(m["id"], api_key))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
