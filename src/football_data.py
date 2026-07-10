"""Thin client for the football-data.org v4 API (match lineups, goals, cards).

Needs a free API key: register at https://www.football-data.org/client/register
and pass it explicitly or set the FOOTBALL_DATA_API_KEY environment variable.
"""
import os
from typing import Optional

import requests

BASE_URL = "https://api.football-data.org/v4"

# Competitions available on the free tier that are relevant to this project.
COMPETITIONS = {
    "WC": "FIFA World Cup",
    "EC": "European Championship",
    "CL": "UEFA Champions League",
    "PL": "Premier League",
}


class FootballDataError(RuntimeError):
    """Raised for API errors (bad key, rate limit, restricted competition/tier)."""


def _request(path: str, api_key: str, params: Optional[dict] = None) -> dict:
    if not api_key:
        raise FootballDataError(
            "No football-data.org API key given. Register for a free one at "
            "https://www.football-data.org/client/register and set "
            "FOOTBALL_DATA_API_KEY, or pass it explicitly."
        )
    resp = requests.get(
        f"{BASE_URL}{path}",
        headers={"X-Auth-Token": api_key},
        params=params,
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()

    try:
        detail = resp.json().get("message", "")
    except ValueError:
        detail = ""

    if resp.status_code == 429:
        raise FootballDataError("football-data.org rate limit hit — wait a minute and try again.")
    if resp.status_code == 403:
        raise FootballDataError(
            detail or "football-data.org says this competition/data isn't available on your API tier."
        )
    if resp.status_code in (400, 401):
        raise FootballDataError(detail or "football-data.org rejected the API key.")
    if resp.status_code == 404:
        raise FootballDataError("football-data.org returned 404 — check the competition/match id.")
    resp.raise_for_status()
    return resp.json()


def api_key_from_env() -> str:
    return os.environ.get("FOOTBALL_DATA_API_KEY", "")


def list_matches(
    competition_code: str,
    api_key: str,
    status: Optional[str] = "FINISHED",
    matchday: Optional[int] = None,
) -> list[dict]:
    """Matches for a competition, most recent first."""
    params = {}
    if status:
        params["status"] = status
    if matchday:
        params["matchday"] = matchday
    data = _request(f"/competitions/{competition_code}/matches", api_key, params)
    matches = data.get("matches", [])
    return sorted(matches, key=lambda m: m["utcDate"], reverse=True)


def get_match(match_id: int, api_key: str) -> dict:
    """Full match detail: score, goals, bookings, substitutions, and — for
    matches football-data.org has lineup data for — each side's starting XI,
    bench and formation."""
    data = _request(f"/matches/{match_id}", api_key)
    return data.get("match", data)
