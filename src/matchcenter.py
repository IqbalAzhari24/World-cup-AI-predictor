"""Player-level match view: first XI, goalscorers, man of the match.

Primary source is a football-data.org /matches/{id} response (see
src/football_data.py). Its free tier doesn't include lineups, goal events or
cards, so when an API-Football key is available (src/api_football.py) and a
matching fixture can be found by team names + date, that fills in the same
detail instead — API-Football's free tier does include it.

Neither provider publishes an official "man of the match" award, so
`man_of_the_match` / `man_of_the_match_from_api_football` derive one from
goals/assists/cards — it's a heuristic, not an official stat.
"""
import argparse

import pandas as pd

from src.api_football import ApiFootballError
from src.api_football import api_key_from_env as api_football_key_from_env
from src.api_football import find_fixture, get_events, get_lineups
from src.football_data import COMPETITIONS, api_key_from_env, get_match, list_matches

GOAL_POINTS = 4
OWN_GOAL_POINTS = -2
ASSIST_POINTS = 2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
WINNING_SIDE_BONUS = 1


def _bump(scores: dict, name: str, team_name: str, points: int):
    if not name:
        return
    entry = scores.setdefault(name, {"name": name, "team": team_name, "points": 0})
    entry["points"] += points


def _best_player(scores: dict, winning_team: str | None) -> dict | None:
    if not scores:
        return None
    if winning_team:
        for entry in scores.values():
            if entry["team"] == winning_team:
                entry["points"] += WINNING_SIDE_BONUS
    return max(scores.values(), key=lambda e: e["points"])


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

    for g in match.get("goals") or []:
        scorer, assist = g.get("scorer") or {}, g.get("assist") or {}
        team_name = (g.get("team") or {}).get("name")
        _bump(scores, scorer.get("name"), team_name, OWN_GOAL_POINTS if g.get("type") == "OWN" else GOAL_POINTS)
        if assist.get("name"):
            _bump(scores, assist["name"], team_name, ASSIST_POINTS)

    for b in match.get("bookings") or []:
        player = b.get("player") or {}
        team_name = (b.get("team") or {}).get("name")
        points = RED_CARD_POINTS if b.get("card") == "RED_CARD" else YELLOW_CARD_POINTS
        _bump(scores, player.get("name"), team_name, points)

    return _best_player(scores, winning_team)


def first_xi_from_api_football(fixture: dict, lineups: list) -> dict[str, dict]:
    """Same shape as first_xi(), sourced from an API-Football lineups response."""
    by_team_id = {l["team"]["id"]: l for l in lineups}
    out = {}
    for side in ("home", "away"):
        team = fixture["teams"][side]
        lineup = by_team_id.get(team["id"])
        players = [
            {
                "shirt": p["player"].get("number"),
                "name": p["player"].get("name"),
                "position": p["player"].get("pos"),
            }
            for p in (lineup.get("startXI") or [])
        ] if lineup else []
        df = pd.DataFrame(players)
        if not df.empty:
            df = df.sort_values("shirt", na_position="last").reset_index(drop=True)
        out[side] = {
            "team": team.get("name"),
            "formation": lineup.get("formation") if lineup else None,
            "lineup": df,
        }
    return out


def goalscorers_from_api_football(events: list) -> pd.DataFrame:
    """Same shape as goalscorers(), sourced from an API-Football events response."""
    rows = []
    for e in events:
        if e.get("type") != "Goal":
            continue
        player, assist = e.get("player") or {}, e.get("assist") or {}
        rows.append(
            {
                "minute": (e.get("time") or {}).get("elapsed"),
                "team": (e.get("team") or {}).get("name"),
                "scorer": player.get("name"),
                "assist": assist.get("name"),
                "type": "OWN" if e.get("detail") == "Own Goal" else e.get("detail"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("minute").reset_index(drop=True)
    return df


def man_of_the_match_from_api_football(fixture: dict, events: list) -> dict | None:
    """Same heuristic as man_of_the_match(), sourced from API-Football events."""
    home_goals = (fixture.get("goals") or {}).get("home")
    away_goals = (fixture.get("goals") or {}).get("away")
    winning_team = None
    if home_goals is not None and away_goals is not None and home_goals != away_goals:
        winning_team = (
            fixture["teams"]["home"]["name"] if home_goals > away_goals else fixture["teams"]["away"]["name"]
        )

    scores: dict[str, dict] = {}
    for e in events:
        team_name = (e.get("team") or {}).get("name")
        detail = e.get("detail") or ""
        if e.get("type") == "Goal":
            player, assist = e.get("player") or {}, e.get("assist") or {}
            _bump(scores, player.get("name"), team_name, OWN_GOAL_POINTS if detail == "Own Goal" else GOAL_POINTS)
            if assist.get("name"):
                _bump(scores, assist["name"], team_name, ASSIST_POINTS)
        elif e.get("type") == "Card":
            player = e.get("player") or {}
            _bump(scores, player.get("name"), team_name, RED_CARD_POINTS if "Red" in detail else YELLOW_CARD_POINTS)

    return _best_player(scores, winning_team)


def enrich_from_api_football(home_name: str, away_name: str, date: str, api_key: str) -> dict | None:
    """Try to find this match on API-Football and pull lineups/goals/MOTM from it.

    Returns {'fixture': ..., 'first_xi': ..., 'goalscorers': ..., 'man_of_the_match': ...},
    or None if no API-Football key is set or no matching fixture is found on
    that date (not an error — most matches simply won't be there, or the date
    matched via team+date lookup differs slightly across providers). Matching
    is by team name + date (see src/api_football.py's find_fixture) since
    football-data.org and API-Football use unrelated ID systems.

    Raises ApiFootballError for a real failure (bad key, rate limit, etc.) —
    callers should let that surface rather than silently falling back, so a
    misconfigured key doesn't look identical to "no data available".
    """
    if not api_key:
        return None
    fixture = find_fixture(home_name, away_name, date, api_key)
    if not fixture:
        return None
    fixture_id = fixture["fixture"]["id"]
    lineups = get_lineups(fixture_id, api_key)
    events = get_events(fixture_id, api_key)

    return {
        "fixture": fixture,
        "first_xi": first_xi_from_api_football(fixture, lineups),
        "goalscorers": goalscorers_from_api_football(events),
        "man_of_the_match": man_of_the_match_from_api_football(fixture, events),
    }


def _print_match(match: dict, api_football_key: str = ""):
    print(match_summary(match))

    enriched = None
    if api_football_key:
        home_name, away_name = match["homeTeam"]["name"], match["awayTeam"]["name"]
        try:
            enriched = enrich_from_api_football(home_name, away_name, match["utcDate"][:10], api_football_key)
        except ApiFootballError as e:
            print(f"\n(API-Football lookup failed: {e})")

    xi = enriched["first_xi"] if enriched else first_xi(match)
    goals = enriched["goalscorers"] if enriched else goalscorers(match)
    motm = enriched["man_of_the_match"] if enriched else man_of_the_match(match)
    if enriched:
        print("(lineups/goals from API-Football)")

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

    print("\nGoals:")
    if goals.empty:
        print("  (none)")
    else:
        for _, row in goals.iterrows():
            assist = f" (assist: {row['assist']})" if pd.notna(row["assist"]) else ""
            print(f"  {row['minute']}' {row['scorer']} — {row['team']}{assist}")

    print("\nMan of the match (heuristic, not an official award):")
    print(f"  {motm['name']} ({motm['team']}) — score {motm['points']}" if motm else "  (not enough match data)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("competition", nargs="?", default="WC", choices=COMPETITIONS,
                         help="Competition code, e.g. WC (World Cup), EC, CL, PL. Default: WC")
    parser.add_argument("--matchday", type=int, help="Only this matchday")
    parser.add_argument("--match-id", type=int, help="Skip the picker, show one match directly")
    parser.add_argument("--api-key", default=None, help="Defaults to FOOTBALL_DATA_API_KEY env var")
    parser.add_argument("--api-football-key", default=None,
                         help="Optional, fills in lineups/goals via API-Football. "
                              "Defaults to API_FOOTBALL_KEY env var")
    args = parser.parse_args()

    api_key = args.api_key or api_key_from_env()
    af_key = args.api_football_key or api_football_key_from_env()

    if args.match_id:
        _print_match(get_match(args.match_id, api_key), af_key)
        return

    matches = list_matches(args.competition, api_key, matchday=args.matchday)
    if not matches:
        print(f"No finished matches found for {COMPETITIONS[args.competition]}.")
        return
    # Each match needs its own request for lineup/goal detail, and the free
    # tier is capped at 10 requests/minute — keep this small by default.
    for m in matches[:5]:
        _print_match(get_match(m["id"], api_key), af_key)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
