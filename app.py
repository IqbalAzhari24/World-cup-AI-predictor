"""Streamlit UI for the World Cup AI predictor.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.data import DATA_PATH, load_matches
from src.elo import match_features
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
    """Horizontal probability bars: (label, probability, css color var name)."""
    html = [STYLE, '<div class="viz-root">']
    for label, prob, color in rows:
        html.append(
            f'<div class="bar-row"><span class="bar-label">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{prob:.1%};background:var(--{color})"></div></div>'
            f'<span class="bar-value">{prob:.1%}</span></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


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

with st.sidebar:
    st.caption(f"Results through **{data_through(DATA_PATH.stat().st_mtime)}**")
    if st.button("🔄 Refresh data & retrain", help="Download the latest results and rebuild the model (~1 min)"):
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

match_tab, tournament_tab, bracket_tab, rankings_tab = st.tabs(
    ["Match predictor", "2026 tournament", "Bracket simulator", "Elo rankings"]
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
