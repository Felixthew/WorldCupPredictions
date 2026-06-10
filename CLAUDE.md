# WorldCupPredictions

Predict the 2026 FIFA World Cup — not just the winner, but the **scoreline of every
individual match** — then simulate the bracket. This is a **skills-refresher project**:
the point is to practice data science / ML and have some fun with the World Cup, not to
ship production code.

## Working style (important)

This is a learning project. Favor teaching over hand-holding:
- **Explain the "why"** behind a modeling or data decision, and the trade-offs, before
  writing code. The reasoning matters more than the output here.
- Prefer **idiomatic, modern pandas / seaborn / scikit-learn** so the patterns are worth
  internalizing. Point out the idiom when it's non-obvious.
- When a step is instructive to write by hand, offer the approach and let the user code it
  rather than dumping a finished cell — but don't be precious about boilerplate.
- Advance **one piece at a time** and check in; the user has been building this
  incrementally and confirming between steps.

## Constraints

- **Python only**, stack limited to: `numpy`, `pandas`, `seaborn` (+ `matplotlib`),
  `scikit-learn`, `scipy`. Keep new dependencies to this set unless asked.
- **Publicly available data only.**
- Small time budget — prefer simple, defensible choices over elaborate ones.

## Environment

- Python 3.13 in `.venv` (`.venv\Scripts\python.exe`). Windows / PowerShell.
- Full stack pinned in `requirements.txt`: pandas 3.0, numpy 2.4, scipy 1.17,
  scikit-learn 1.9, seaborn 0.13, matplotlib 3.10, jupyterlab. Install with
  `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Layout

| Path | What it is |
|---|---|
| `predictor.ipynb` | The working notebook — primary surface for EDA, viz, modeling. |
| `predictor.py` | Library version of the pipeline; functions mirror the notebook cells, `main()` runs the whole chain and returns `(df_with_stats, final_ratings)`. Keep the two in sync. |
| `results.csv` | ~49k international match results back to 1872 (date, teams, scores, tournament, city, country, neutral). The spine of the project. |
| `male_teams_14-24.csv` | Stefano Leone's FIFA 15–24 team ratings. `league_id == 78` ("Friendly International") selects national teams. |
| `sofifa_teams_fc25.html`, `sofifa_teams_fc26.html` | Manually-saved sofifa listing pages for FC25/FC26 (sofifa is behind Cloudflare — HTTP scraping is blocked, hence the manual save). |

## Feature pipeline (order matters)

ELO is computed over **full history**, everything else on the **2021+ slice**:

1. **ELO** — eloratings.net-style, with `k_factor` by match importance and `g_multiplier`
   for goal difference. Stamps `elo_home_pre` / `elo_away_pre` (pre-match, no leakage).
2. **Recent goals** — last-10-game GF/GA, **residualised against opponent ELO** (a goal vs
   a minnow ≠ a goal vs a top side). Centered near 0; >0 means out-performing the matchups.
3. **Recent W/D/L** — last-5-game points (3/1/0).
4. **Rest days** — days since each team's previous match.
5. **Host-continent advantage** — flags whether each team shares a confederation with the
   host (re-reads `results.csv` for the dropped `country` column).
6. **sofifa team stats** — overall/attack/midfield/defence/starting-XI-age, joined via
   `merge_asof` on actual rating-release dates so a match only ever sees the snapshot that
   existed at kickoff.

## Conventions & gotchas

- **No leakage**: every feature is computed from data strictly *before* the match. Preserve
  this when adding features — walk matches in date order and stamp pre-match values.
- **Team-name normalization** across sources is fiddly (Czechia↔Czech Republic,
  Türkiye↔Turkey, Korea Republic↔South Korea, Côte d'Ivoire↔Ivory Coast, …). Maps live in
  `predictor.py` (`SOFIFA_TO_CANONICAL`, `CSV_TO_CANONICAL`). Reuse them.
- **FC25 scrape is incomplete** (~30 teams missing, incl. Brazil — that's all sofifa had).
  Those are back-filled from FC24 ratings in `build_team_stats`.
- **Tactic-style data** (build-up, defensive approach, whole-team age) stops after FC24 —
  intentionally dropped.
- **NaN team-stats rows** (small nations EA never rated) are a known open issue, deferred to
  the modeling step (drop vs impute — decide from the actual NaN count).

## Visualizations (notebook "Visualizations" + "Tournament Stats" sections)

Built so far: goals-per-side histogram with optional Poisson overlay
(`plot_goal_distribution(overlay_poisson=)`); scoreline-probability heatmap; FC26+ELO
clustermap of the 48 WC teams; goals-by-tournament boxenplot + ECDF; ELO unpredictability
index by competition; heavy-vs-slight-favourite upset-rate; Poisson dispersion report.
Helpers: `tournament_group()` buckets the 200 raw tournament strings into 5 (World Cup /
Continental Cups / Qualifiers / Friendlies / Other); `WC_2026_TEAMS` (top cell) lists the
48 participants.

Findings worth not re-deriving: goal scoring is **remarkably scale-invariant** across
competitions (ECDF/boxen near-identical); ELO unpredictability runs Qualifiers ≈0.26 (most
predictable, big mismatches) → World Cup ≈0.36 (least), but cross-tournament differences are
mostly within noise — the upset/calibration thread bottomed out, calibration deferred to the
model-eval step where it's load-bearing.

Viz gotchas:
- **Inline backend closes figures at the end of each cell** — you can't build a figure in one
  cell and `ax.plot` onto it in the next. Wrap plots in a function that makes a fresh figure.
- **seaborn `x`/`y`/`hue` need long-form data**; a grouped Series is wide-form → `reset_index()`
  first. **Error bars need the raw rows, not pre-aggregated means** (seaborn computes mean+CI
  itself; feed it the per-match frame).
- **`PercentFormatter(xmax=1)`** for percent-formatted colorbars/axes when values are 0–1.
- **`sns.clustermap` caches the scipy check at import** — if it errors "requires scipy", restart
  the kernel.

## Where this is headed

Model = **two independent goal-count regressions** (home goals, away goals), combined by the
**outer product of their two Poisson PMFs** for the joint scoreline grid (Maher-style).
Chronological train/test split; preserve no-leakage.

- **Poisson check (done, marginal):** goals are **overdispersed** — var/mean ≈ 1.60 home /
  1.44 away / 1.54 neutral. But that's the *marginal*, where a mixture of many per-match λs
  inflates variance even if each match is Poisson. The decision-relevant test is **conditional**
  dispersion: residual dispersion after fitting, or a homogeneous-ELO-slice proxy. **Open.**
- **Model choice:** start with `PoissonRegressor` (log-link GLM; **`alpha` defaults to 1.0**, i.e.
  regularized, not 0). If conditional overdispersion survives, swap to
  `TweedieRegressor(power≈1.5, link="log")` — same GLM, Poisson is the `p=1` special case.
  (Negative Binomial is the natural discrete overdispersed distribution but isn't in sklearn.)
- **Independence caveat:** two independent Poissons assume home/away goals are conditionally
  independent. Real dependence is small and concentrated in low scores (Dixon-Coles). Diagnostic:
  observed scoreline heatmap vs outer-product-of-marginals. DC τ correction deferred (out of budget).
- **Evaluate:** RMSE on goals, W/D/L accuracy, Brier; calibration/reliability curve here.
- **Then:** Monte Carlo of the 48-team tournament + seaborn viz.

## Running

```powershell
.venv\Scripts\python.exe predictor.py   # runs the full feature pipeline, prints ELO top 15
```
Or work interactively in `predictor.ipynb` (jupyterlab).
