"""Thin client for API-Football (api-sports.io).

Used only to fill in lineups, goal events and cards for a match already found
via football-data.org — that API's free tier doesn't include player data (see
src/football_data.py), and API-Football's free tier does.

Needs a free API key: register at https://dashboard.api-football.com/register
(no credit card) and set API_FOOTBALL_KEY, or pass it explicitly.
"""
import os
from typing import Optional

import requests

from src._ratelimit import Throttle

BASE_URL = "https://v3.football.api-sports.io"

# Free tier allows 10 requests/minute; margin keeps a shared key from tipping
# over that when several visitors use the app at once.
_throttle = Throttle(min_interval=6.5)


class ApiFootballError(RuntimeError):
    """Raised for API-Football errors (bad key, rate limit, bad response)."""


def api_key_from_env() -> str:
    return os.environ.get("API_FOOTBALL_KEY", "")


def _request(path: str, api_key: str, params: Optional[dict] = None) -> list:
    if not api_key:
        raise ApiFootballError(
            "No API-Football key given. Register for a free one at "
            "https://dashboard.api-football.com/register and set "
            "API_FOOTBALL_KEY, or pass it explicitly."
        )
    try:
        _throttle.wait()
        resp = requests.get(
            f"{BASE_URL}{path}",
            headers={"x-apisports-key": api_key},
            params=params,
            timeout=15,
        )
    except requests.RequestException as e:
        raise ApiFootballError(f"Could not reach API-Football: {e}") from e

    if resp.status_code == 429:
        raise ApiFootballError(
            "API-Football rate limit hit (100 requests/day on the free tier) — try again later."
        )
    try:
        data = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise ApiFootballError(f"API-Football returned an unreadable response (status {resp.status_code}).")

    errors = data.get("errors")
    if errors:
        msg = "; ".join(str(v) for v in errors.values()) if isinstance(errors, dict) else str(errors)
        raise ApiFootballError(f"API-Football error: {msg}")
    if resp.status_code != 200:
        raise ApiFootballError(f"API-Football returned HTTP {resp.status_code}.")

    return data.get("response", [])


def _normalize_team_name(name: str) -> str:
    name = name.lower().replace(".", "").replace("-", " ")
    name = name.replace(" fc", "").replace("fc ", "")
    return " ".join(name.split())


def find_fixture(home_name: str, away_name: str, date: str, api_key: str) -> Optional[dict]:
    """Look up a fixture by team names and match date (YYYY-MM-DD).

    Matches loosely (case-insensitive, substring) since club/country naming
    conventions differ slightly between football-data.org and API-Football
    (e.g. "Paris Saint-Germain FC" vs "Paris Saint Germain"). Returns the raw
    /fixtures response item for the match, or None if no fixture is found on
    that date with both team names appearing.
    """
    fixtures = _request("/fixtures", api_key, params={"date": date})
    home_n, away_n = _normalize_team_name(home_name), _normalize_team_name(away_name)
    for fx in fixtures:
        fh = _normalize_team_name(fx["teams"]["home"]["name"])
        fa = _normalize_team_name(fx["teams"]["away"]["name"])
        if (home_n in fh or fh in home_n) and (away_n in fa or fa in away_n):
            return fx
    return None


def get_lineups(fixture_id: int, api_key: str) -> list:
    return _request("/fixtures/lineups", api_key, params={"fixture": fixture_id})


def get_events(fixture_id: int, api_key: str) -> list:
    return _request("/fixtures/events", api_key, params={"fixture": fixture_id})
