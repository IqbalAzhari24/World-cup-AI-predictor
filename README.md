# World Cup AI Predictor ⚽

A lightweight but powerful AI that predicts international football matches and
simulates World Cup brackets. No GPU, no deep learning, no API keys — just
well-engineered features and gradient boosting, which is the sweet spot for
this problem size.

## How it works

1. **Data** — ~48,000 international matches since 1872 from the open
   [martj42/international_results](https://github.com/martj42/international_results)
   dataset (auto-downloaded on first run, includes upcoming fixtures).
2. **Features** (`src/elo.py`) — computed chronologically with zero leakage:
   - **Elo ratings** with tournament-weighted K-factor (World Cup > qualifiers > friendlies)
     and margin-of-victory scaling — the single strongest signal in football.
   - Home advantage (+60 Elo, disabled on neutral ground).
   - Recent form, attack and defense averages over the last 5 matches.
3. **Model** (`src/model.py`) — scikit-learn `HistGradientBoostingClassifier`
   producing calibrated win/draw/loss probabilities.
4. **Tournament simulation** (`src/simulate.py`) — Monte Carlo: play out the
   knockout bracket 10,000 times to get each team's chance of lifting the cup.

**Current performance** (held-out last 4 years, ~4,000 matches):
60.7% three-way accuracy, log loss 0.871 (vs 1.053 naive baseline) —
on par with published football-prediction models.

## Quick start

```bash
pip install -r requirements.txt

# Web UI — match predictor, bracket simulator, Elo rankings
streamlit run app.py

# 1. Download data + train (takes ~1 minute)
python -m src.model

# 2. Predict a match (neutral venue by default, like a World Cup)
python -m src.predict "France" "Morocco"

# 3. Simulate a knockout bracket (teams in bracket order)
python -m src.simulate "Argentina" "Egypt" "Switzerland" "Colombia" \
                       "France" "Morocco" "Brazil" "Spain" -n 10000

# 4. Simulate the FULL 2026 World Cup from the group stage
python -m src.tournament -n 10000
```

Example output:

```
France vs Morocco  (neutral venue)
  France win  :  49.0%
  Draw        :  26.9%
  Morocco win :  24.1%
```

## Project layout

```
app.py         # Streamlit web UI                        (streamlit run app.py)
src/
  data.py      # dataset download + loading (played matches & upcoming fixtures)
  elo.py       # Elo ratings + rolling form, computed match-by-match
  model.py     # training, evaluation, artifact saving   (python -m src.model)
  predict.py   # single-match CLI                        (python -m src.predict)
  simulate.py  # Monte Carlo knockout simulation         (python -m src.simulate)
  tournament.py# full tournament: groups + knockout      (python -m src.tournament)
```

The 2026 groups aren't hardcoded — `tournament.py` recovers them from the
fixture list by finding the 4-team connected components of the group-stage
match graph. Group games sample full scorelines so points, goal difference
and goals-for drive the standings like the real tiebreakers; the top 2 per
group plus the 8 best thirds are seeded into the round-of-32 bracket.

## Roadmap

- [x] Group-stage simulation (points, goal difference, tiebreakers) for full-tournament odds
- [ ] Poisson goal model for exact scoreline probabilities
- [ ] Probability calibration (isotonic) + Brier score tracking
- [x] Streamlit web UI (match predictor, bracket simulator, Elo rankings)
- [ ] Auto-refresh data weekly and re-train
