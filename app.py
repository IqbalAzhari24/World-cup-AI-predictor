"""Streamlit UI for the World Cup AI predictor.  Run:  streamlit run app.py"""
import html
import time

import pandas as pd
import streamlit as st

from src.api_football import ApiFootballError
from src.api_football import api_key_from_env as api_football_key_from_env
from src.data import DATA_PATH, load_matches
from src.elo import match_features
from src.football_data import COMPETITIONS, FootballDataError, api_key_from_env, get_match, list_matches
from src.matchcenter import enrich_from_api_football, first_xi, goalscorers, man_of_the_match, match_summary
from src.model import FEATURES, load_artifacts, train
from src.simulate import simulate_knockout
from src.tournament import extract_groups, simulate_tournament

st.set_page_config(page_title="World Cup AI Predictor", page_icon="⚽", layout="centered")

# Validated palette (light / dark), applied via prefers-color-scheme
STYLE = """
<style>
.viz-root {
  --home:  #2a78d6;
  --draw:  #898781;
  --away:  #e34948;
  --champ: #2a78d6;
  --track: #e1e0d9;
  --ink:   #0b0b0b;
  --ink-2: #52514e;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    --home:  #3987e5;
    --away:  #e66767;
    --champ: #3987e5;
    --track: #2c2c2a;
    --ink:   #ffffff;
    --ink-2: #c3c2b7;
  }
}
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.bar-row  { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
.bar-label { flex: 0 0 9em; color: var(--ink); font-size: 0.9rem;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; height: 8px; background: var(--track); border-radius: 4px; }
.bar-fill  { height: 100%; border-radius: 4px; min-width: 2px; }
.bar-value { flex: 0 0 4em; text-align: right; color: var(--ink-2);
             font-size: 0.9rem; font-variant-numeric: tabular-nums; }
</style>
"""


def bar_chart(rows: list[tuple[str, float, str]]):
    """Horizontal probability bars: (label, probability, css color var name).

    Labels are HTML-escaped before going into unsafe_allow_html markup —
    several come from external data (team/player names from football-data.org
    and the open match-results dataset), not something we fully control.
    """
    parts = [STYLE, '<div class="viz-root">']
    for label, prob, color in rows:
        parts.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(str(label))}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{prob:.1%};background:var(--{color})"></div></div>'
            f'<span class="bar-value">{prob:.1%}</span></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training model on 48k international matches (first run only)...")
def get_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return train(verbose=False)


@st.cache_data
def data_through(_mtime: float) -> str:
    """Date of the latest played match in the local dataset."""
    df = pd.read_csv(DATA_PATH, usecols=["date", "home_score"])
    return df.loc[df.home_score.notna(), "date"].max()


model, states = get_artifacts()
teams_by_rating = sorted(states, key=lambda t: states[t].rating, reverse=True)

# Refreshing retrains the model for every visitor at once (Streamlit shares one
# process across all of them), so a cooldown stops the button being smashed
# into a repeated free-for-all retrain. DATA_PATH's mtime is shared disk state,
# not per-session, so this cools down for everyone rather than per-browser.
REFRESH_COOLDOWN_SECONDS = 15 * 60

with st.sidebar:
    st.caption(f"Results through **{data_through(DATA_PATH.stat().st_mtime)}**")
    cooldown_left = REFRESH_COOLDOWN_SECONDS - (time.time() - DATA_PATH.stat().st_mtime)
    if cooldown_left > 0:
        st.button(
            "🔄 Refresh data & retrain",
            disabled=True,
            help=f"Available again in ~{int(cooldown_left // 60) + 1} min "
                 "(shared cooldown across everyone using this app).",
        )
    elif st.button("🔄 Refresh data & retrain", help="Download the latest results and rebuild the model (~1 min)"):
        with st.spinner("Downloading latest results and retraining..."):
            load_matches(refresh=True)
            train(verbose=False)
        get_artifacts.clear()
        st.cache_data.clear()
        st.rerun()

st.title("⚽ World Cup AI Predictor")
st.caption(
    "Elo ratings + gradient boosting over 48,000 international matches. "
    "Probabilities, not prophecies."
)

match_tab, tournament_tab, bracket_tab, rankings_tab, matchcenter_tab = st.tabs(
    ["Match predictor", "2026 tournament", "Bracket simulator", "Elo rankings", "Match center"]
)

with match_tab:
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams_by_rating, index=teams_by_rating.index("France"))
    team_b = col2.selectbox("Team B", teams_by_rating, index=teams_by_rating.index("Morocco"))
    neutral = not st.toggle(f"{team_a} plays at home (off = neutral venue, like a World Cup)")

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        row = match_features(states[team_a], states[team_b], neutral)
        p_win, p_draw, p_loss = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
        st.subheader(f"{team_a} vs {team_b}")
        bar_chart([(f"{team_a} win", p_win, "home"),
                   ("Draw", p_draw, "draw"),
                   (f"{team_b} win", p_loss, "away")])

        try:
            from src.scoreline import top_scorelines
            (xg_a, xg_b), lines = top_scorelines(team_a, team_b, neutral)
        except FileNotFoundError:
            lines = None  # artifact predates the goal models; refresh & retrain to enable
        if lines:
            st.subheader("Most likely scorelines")
            st.caption(f"Expected goals: {team_a} **{xg_a:.2f}** — **{xg_b:.2f}** {team_b}")
            bar_chart([(f"{gh}-{ga}", p, "champ") for (gh, ga), p in lines])

with tournament_tab:
    st.markdown(
        "Simulates the **entire 2026 World Cup** from the group stage: round-robin "
        "groups with sampled scorelines (points, goal difference, tiebreakers), "
        "top 2 + 8 best thirds advance, seeded round-of-32 knockout."
    )
    n_full_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000, key="full_sims")

    @st.cache_data(show_spinner="Simulating tournaments...")
    def tournament_odds(n_sims: int):
        champions, finalists = simulate_tournament(2026, n_sims, artifacts=(model, states))
        return champions, finalists

    if st.button("Simulate World Cup 2026", type="primary"):
        champions, finalists = tournament_odds(n_full_sims)
        st.subheader("Chance of winning the cup")
        bar_chart([(team, wins / n_full_sims, "champ")
                   for team, wins in champions.most_common(15)])
        st.subheader("Chance of reaching the final")
        bar_chart([(team, finalists[team] / n_full_sims, "champ")
                   for team, _ in champions.most_common(15)])
        with st.expander("The groups (recovered from the fixture list)"):
            for i, g in enumerate(extract_groups(2026)):
                st.markdown(f"**Group {chr(65 + i)}** — {', '.join(g)}")

with bracket_tab:
    st.markdown(
        "Pick teams **in bracket order** — 1 plays 2, 3 plays 4, winners meet. "
        "Team count must be a power of two."
    )
    default_bracket = ["Argentina", "Egypt", "Switzerland", "Colombia",
                       "France", "Morocco", "Brazil", "Spain"]
    bracket = st.multiselect("Bracket", teams_by_rating,
                             default=[t for t in default_bracket if t in states])
    n_sims = st.slider("Simulations", 1000, 20000, 10000, step=1000)

    n = len(bracket)
    if n < 2 or n & (n - 1):
        st.info("Select a power of two number of teams (2, 4, 8, 16, ...).")
    elif st.button("Simulate tournament", type="primary"):
        champs = simulate_knockout(bracket, n_sims, artifacts=(model, states))
        st.subheader("Chance of winning the cup")
        bar_chart([(team, wins / n_sims, "champ") for team, wins in champs.most_common()])

with matchcenter_tab:
    st.markdown(
        "Player-level detail — starting XI, goalscorers, and a heuristic "
        "**Man of the Match** — for real matches, pulled live from "
        "[football-data.org](https://www.football-data.org/)."
    )
    api_key = api_key_from_env()
    comp_code = st.selectbox(
        "Competition", list(COMPETITIONS), format_func=lambda c: f"{COMPETITIONS[c]} ({c})"
    )

    @st.cache_data(show_spinner="Fetching finished matches...", ttl=300)
    def cached_matches(competition: str, key: str):
        return list_matches(competition, key)

    @st.cache_data(show_spinner="Fetching match detail...", ttl=300)
    def cached_match(match_id: int, key: str):
        return get_match(match_id, key)

    if not api_key:
        st.info(
            "Match center needs a football-data.org API key. Set "
            "`FOOTBALL_DATA_API_KEY` (env var, or `.streamlit/secrets.toml` "
            "for a deployed app) — see the README."
        )
    else:
        try:
            matches = cached_matches(comp_code, api_key)
        except FootballDataError as e:
            matches = None
            st.error(str(e))

        if matches is not None:
            if not matches:
                st.warning(f"No finished matches found for {COMPETITIONS[comp_code]}.")
            else:
                choice = st.selectbox(
                    "Match", matches, format_func=lambda m: f"{m['homeTeam']['name']} vs "
                    f"{m['awayTeam']['name']} ({m['utcDate'][:10]})"
                )
                try:
                    match = cached_match(choice["id"], api_key)
                except FootballDataError as e:
                    match = None
                    st.error(str(e))

                if match is not None:
                    st.subheader(match_summary(match))

                    home_name = match["homeTeam"]["name"]
                    away_name = match["awayTeam"]["name"]
                    st.markdown("**Our prediction vs the actual result**")
                    if home_name in states and away_name in states:
                        row = match_features(states[home_name], states[away_name], neutral=True)
                        p_win, p_draw, p_loss = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
                        bar_chart([(f"{home_name} win", p_win, "home"),
                                   ("Draw", p_draw, "draw"),
                                   (f"{away_name} win", p_loss, "away")])

                        full = match.get("score", {}).get("fullTime", {})
                        h, a = full.get("home"), full.get("away")
                        if match.get("status") == "FINISHED" and h is not None and a is not None:
                            actual = f"{home_name} win" if h > a else f"{away_name} win" if a > h else "Draw"
                            predicted = max(
                                [(f"{home_name} win", p_win), ("Draw", p_draw), (f"{away_name} win", p_loss)],
                                key=lambda x: x[1],
                            )[0]
                            mark = "✅ matched our top pick" if predicted == actual else "❌ missed our top pick"
                            st.caption(f"Actual: **{actual}** ({h}-{a}). {mark} (**{predicted}**).")
                        else:
                            st.caption("Match hasn't kicked off yet — showing our prediction only.")
                    else:
                        st.caption(
                            "Prediction unavailable — our model only covers international "
                            "teams (World Cup, Euros), not club sides."
                        )

                    af_key = api_football_key_from_env()

                    @st.cache_data(show_spinner="Checking API-Football for lineups/goals...", ttl=3600)
                    def cached_af_enrichment(home: str, away: str, date: str, key: str):
                        return enrich_from_api_football(home, away, date, key)

                    enrichment = None
                    af_error = None
                    if af_key:
                        try:
                            enrichment = cached_af_enrichment(home_name, away_name, match["utcDate"][:10], af_key)
                        except ApiFootballError as e:
                            af_error = str(e)

                    xi = enrichment["first_xi"] if enrichment else first_xi(match)
                    goals = enrichment["goalscorers"] if enrichment else goalscorers(match)
                    motm = enrichment["man_of_the_match"] if enrichment else man_of_the_match(match)

                    col1, col2 = st.columns(2)
                    for col, side in ((col1, "home"), (col2, "away")):
                        info = xi[side]
                        title = info["team"] or side.title()
                        if info["formation"]:
                            title += f" — {info['formation']}"
                        col.markdown(f"**{title}**")
                        if info["lineup"].empty:
                            col.caption("No lineup data available for this match.")
                        else:
                            col.dataframe(
                                info["lineup"].rename(
                                    columns={"shirt": "#", "name": "Name", "position": "Position"}
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )

                    st.markdown("**Goals**")
                    if goals.empty:
                        st.caption("No goals in this match.")
                    else:
                        st.dataframe(
                            goals.rename(columns={
                                "minute": "Min", "team": "Team", "scorer": "Scorer",
                                "assist": "Assist", "type": "Type",
                            }),
                            hide_index=True,
                            use_container_width=True,
                        )

                    st.markdown("**Man of the match**")
                    if motm:
                        st.write(f"🏅 **{motm['name']}** ({motm['team']}) — heuristic score {motm['points']}")
                        st.caption(
                            "Not an official award — no provider publishes one. "
                            "Derived from goals, assists and cards in this match."
                        )
                    else:
                        st.caption("Not enough match event data to compute one.")

                    if enrichment:
                        st.caption("Lineups and goals above are from API-Football (matched by team + date).")
                    elif af_error:
                        st.caption(f"API-Football lookup failed: {af_error}")
                    elif af_key:
                        st.caption(
                            "No matching API-Football fixture found for this team/date pairing — "
                            "showing football-data.org data only."
                        )
                    else:
                        st.caption(
                            "Tip: set `API_FOOTBALL_KEY` to fill in lineups/goals that "
                            "football-data.org's free tier doesn't include — see the README."
                        )

with rankings_tab:
    top = st.slider("Show top", 10, 50, 25, step=5)
    st.dataframe(
        pd.DataFrame(
            {
                "Team": teams_by_rating[:top],
                "Elo rating": [round(states[t].rating) for t in teams_by_rating[:top]],
                "Form (pts/game, last 5)": [round(states[t].form(), 2) for t in teams_by_rating[:top]],
            },
            index=range(1, top + 1),
        ),
        use_container_width=True,
    )
