import re
from collections import deque

import numpy as np
import pandas as pd


# === ELO ==================================================================

def k_factor(tournament: str) -> float:
    """ELO weight by match importance (modeled on eloratings.net)."""
    t = tournament.lower()
    if 'world cup' in t and 'qualif' not in t:
        return 60.0
    if t in {'copa america', 'uefa euro', 'african cup of nations', 'afc asian cup'}:
        return 50.0
    if 'qualif' in t or 'nations league' in t:
        return 40.0
    if t == 'friendly':
        return 20.0
    return 30.0


def g_multiplier(goal_diff: int) -> float:
    """Goal-difference multiplier — bigger wins move ELO more."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(r_home: float, r_away: float, home_adv: float = 100.0) -> float:
    """Probability the home team takes the points (draw counts as 0.5)."""
    return 1.0 / (1.0 + 10.0 ** (-(r_home - r_away + home_adv) / 400.0))


def update_match(r_home, r_away, home_score, away_score, tournament, neutral):
    """Apply one match. Returns the new (home_rating, away_rating)."""
    h = 0.0 if neutral else 100.0
    exp_home = expected_score(r_home, r_away, h)

    if home_score > away_score:
        w_home = 1.0
    elif home_score < away_score:
        w_home = 0.0
    else:
        w_home = 0.5

    k = k_factor(tournament) * g_multiplier(home_score - away_score)
    delta = k * (w_home - exp_home)
    return r_home + delta, r_away - delta


def compute_elo_history(matches: pd.DataFrame, initial_rating: float = 1500.0):
    """Walk matches in date order; stamp each row with pre-match ELO for both teams.

    Returns (df_with_elo, final_ratings_dict). Run over the FULL history
    (not a recent slice) so ratings have time to separate teams.
    """
    matches = matches.sort_values('date', kind='mergesort').reset_index(drop=True)
    ratings: dict[str, float] = {}
    elo_home_pre = np.empty(len(matches))
    elo_away_pre = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, initial_rating)
        r_away = ratings.get(away, initial_rating)
        elo_home_pre[i] = r_home
        elo_away_pre[i] = r_away

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        ratings[home], ratings[away] = update_match(
            r_home, r_away,
            int(row.home_score), int(row.away_score),
            row.tournament, bool(row.neutral),
        )

    matches = matches.copy()
    matches['elo_home_pre'] = elo_home_pre
    matches['elo_away_pre'] = elo_away_pre
    return matches, ratings


# === Recent goals (residualised vs opponent ELO) ==========================

def compute_recent_goals(matches: pd.DataFrame) -> pd.DataFrame:
    """Stamp each row with last-10-games GF and GA, NORMALISED against opp ELO.

    Fits `expected_goals_scored = a + b*opp_elo` on completed matches (both
    perspectives, 2 obs per match), then stores residuals per game and
    averages them over each team's last-10-game window:
        > 0 → out-scoring what an avg team would manage vs those opponents
        < 0 → underperforming for the matchups faced
    Same baseline covers GA by symmetry (goals conceded by team = goals
    scored by opp, where opp's `opp_elo` from their POV is this team's elo).
    """
    recent = 10
    matches = matches.sort_values('date', kind='mergesort').reset_index(drop=True)

    played = matches.dropna(subset=['home_score', 'away_score'])
    opp_elo = np.concatenate([played['elo_away_pre'].values,
                              played['elo_home_pre'].values])
    scored = np.concatenate([played['home_score'].astype(float).values,
                             played['away_score'].astype(float).values])
    b, a = np.polyfit(opp_elo, scored, 1)
    print(f"baseline: expected_goals_scored = {a:.3f} + {b:.6f} * opp_elo")

    exp_home_scored = a + b * matches['elo_away_pre'].values  # home's expected GF
    exp_away_scored = a + b * matches['elo_home_pre'].values  # away's expected GF

    history: dict[str, deque] = {}
    gf_home_recent = np.empty(len(matches))
    ga_home_recent = np.empty(len(matches))
    gf_away_recent = np.empty(len(matches))
    ga_away_recent = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        dq_home = history.setdefault(home, deque(maxlen=recent))
        dq_away = history.setdefault(away, deque(maxlen=recent))

        gf_home_recent[i], ga_home_recent[i] = mean_resid(dq_home)
        gf_away_recent[i], ga_away_recent[i] = mean_resid(dq_away)

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        dq_home.append((
            float(row.home_score) - exp_home_scored[i],
            float(row.away_score) - exp_away_scored[i],
        ))
        dq_away.append((
            float(row.away_score) - exp_away_scored[i],
            float(row.home_score) - exp_home_scored[i],
        ))

    matches = matches.copy()
    matches['gf_home_recent'] = gf_home_recent
    matches['ga_home_recent'] = ga_home_recent
    matches['gf_away_recent'] = gf_away_recent
    matches['ga_away_recent'] = ga_away_recent
    return matches


def mean_resid(dq):
    """Mean (gf_residual, ga_residual) over a team's last-N games. Empty → 0."""
    if not dq:
        return 0.0, 0.0
    n = len(dq)
    gf = sum(g for g, _ in dq) / n
    ga = sum(c for _, c in dq) / n
    return gf, ga


# === Recent W/D/L points (last 5) =========================================

def compute_recent_wdl(matches: pd.DataFrame) -> pd.DataFrame:
    """Last-5-games points (3 for W, 1 for D, 0 for L) per team."""
    recent_games = 5
    matches = matches.sort_values('date', kind='mergesort').reset_index(drop=True)
    recent_wdl: dict[str, deque] = {}

    wdl_home_recent = np.empty(len(matches))
    wdl_away_recent = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        wdl_home = recent_wdl.setdefault(home, deque(maxlen=recent_games))
        wdl_away = recent_wdl.setdefault(away, deque(maxlen=recent_games))

        wdl_home_recent[i] = calc_wdl(wdl_home)
        wdl_away_recent[i] = calc_wdl(wdl_away)

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        wdl_home.append((row.home_score, row.away_score))
        wdl_away.append((row.away_score, row.home_score))

    matches = matches.copy()
    matches['wdl_home_recent'] = wdl_home_recent
    matches['wdl_away_recent'] = wdl_away_recent
    return matches


def calc_wdl(wdl_team) -> int:
    wdl = 0
    for match in wdl_team:
        if match[0] > match[1]:
            wdl += 3
        elif match[0] == match[1]:
            wdl += 1
    return wdl


# === Rest days ============================================================

def compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous international match. NaN on first appearance."""
    matches = matches.sort_values('date', kind='mergesort').reset_index(drop=True)
    last_match: dict[str, pd.Timestamp] = {}
    rest_home = np.full(len(matches), np.nan)
    rest_away = np.full(len(matches), np.nan)

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        if home in last_match:
            rest_home[i] = (row.date - last_match[home]).days
        if away in last_match:
            rest_away[i] = (row.date - last_match[away]).days

        last_match[home] = row.date
        last_match[away] = row.date

    matches = matches.copy()
    matches['rest_home'] = rest_home
    matches['rest_away'] = rest_away
    return matches


# === Host-continent advantage =============================================

CONTINENT = {
    # UEFA
    'Albania': 'EU', 'Andorra': 'EU', 'Armenia': 'EU', 'Austria': 'EU', 'Azerbaijan': 'EU',
    'Belarus': 'EU', 'Belgium': 'EU', 'Bosnia and Herzegovina': 'EU', 'Bulgaria': 'EU',
    'Croatia': 'EU', 'Cyprus': 'EU', 'Czech Republic': 'EU', 'Czechia': 'EU', 'Denmark': 'EU',
    'England': 'EU', 'Estonia': 'EU', 'Faroe Islands': 'EU', 'Finland': 'EU', 'France': 'EU',
    'Georgia': 'EU', 'Germany': 'EU', 'Gibraltar': 'EU', 'Greece': 'EU', 'Hungary': 'EU',
    'Iceland': 'EU', 'Israel': 'EU', 'Italy': 'EU', 'Kazakhstan': 'EU', 'Kosovo': 'EU',
    'Latvia': 'EU', 'Liechtenstein': 'EU', 'Lithuania': 'EU', 'Luxembourg': 'EU', 'Malta': 'EU',
    'Moldova': 'EU', 'Montenegro': 'EU', 'Netherlands': 'EU', 'North Macedonia': 'EU',
    'Northern Ireland': 'EU', 'Norway': 'EU', 'Poland': 'EU', 'Portugal': 'EU',
    'Republic of Ireland': 'EU', 'Romania': 'EU', 'Russia': 'EU', 'San Marino': 'EU',
    'Scotland': 'EU', 'Serbia': 'EU', 'Slovakia': 'EU', 'Slovenia': 'EU', 'Spain': 'EU',
    'Sweden': 'EU', 'Switzerland': 'EU', 'Turkey': 'EU', 'Ukraine': 'EU', 'Wales': 'EU',
    # CONMEBOL
    'Argentina': 'SA', 'Bolivia': 'SA', 'Brazil': 'SA', 'Chile': 'SA', 'Colombia': 'SA',
    'Ecuador': 'SA', 'Paraguay': 'SA', 'Peru': 'SA', 'Uruguay': 'SA', 'Venezuela': 'SA',
    # CONCACAF
    'Antigua and Barbuda': 'NA', 'Aruba': 'NA', 'Bahamas': 'NA', 'Barbados': 'NA',
    'Belize': 'NA', 'Bermuda': 'NA', 'British Virgin Islands': 'NA', 'Canada': 'NA',
    'Cayman Islands': 'NA', 'Costa Rica': 'NA', 'Cuba': 'NA', 'Curaçao': 'NA', 'Dominica': 'NA',
    'Dominican Republic': 'NA', 'El Salvador': 'NA', 'Grenada': 'NA', 'Guatemala': 'NA',
    'Guyana': 'NA', 'Haiti': 'NA', 'Honduras': 'NA', 'Jamaica': 'NA', 'Martinique': 'NA',
    'Mexico': 'NA', 'Montserrat': 'NA', 'Nicaragua': 'NA', 'Panama': 'NA', 'Puerto Rico': 'NA',
    'Saint Kitts and Nevis': 'NA', 'Saint Lucia': 'NA',
    'Saint Vincent and the Grenadines': 'NA', 'Sint Maarten': 'NA', 'Suriname': 'NA',
    'Trinidad and Tobago': 'NA', 'Turks and Caicos Islands': 'NA', 'United States': 'NA',
    'US Virgin Islands': 'NA',
    # CAF
    'Algeria': 'AF', 'Angola': 'AF', 'Benin': 'AF', 'Botswana': 'AF', 'Burkina Faso': 'AF',
    'Burundi': 'AF', 'Cameroon': 'AF', 'Cape Verde': 'AF', 'Central African Republic': 'AF',
    'Chad': 'AF', 'Comoros': 'AF', 'Congo': 'AF', 'DR Congo': 'AF', 'Djibouti': 'AF',
    'Egypt': 'AF', 'Equatorial Guinea': 'AF', 'Eritrea': 'AF', 'Eswatini': 'AF',
    'Ethiopia': 'AF', 'Gabon': 'AF', 'Gambia': 'AF', 'Ghana': 'AF', 'Guinea': 'AF',
    'Guinea-Bissau': 'AF', 'Ivory Coast': 'AF', 'Kenya': 'AF', 'Lesotho': 'AF',
    'Liberia': 'AF', 'Libya': 'AF', 'Madagascar': 'AF', 'Malawi': 'AF', 'Mali': 'AF',
    'Mauritania': 'AF', 'Mauritius': 'AF', 'Morocco': 'AF', 'Mozambique': 'AF',
    'Namibia': 'AF', 'Niger': 'AF', 'Nigeria': 'AF', 'Rwanda': 'AF',
    'São Tomé and Príncipe': 'AF', 'Senegal': 'AF', 'Seychelles': 'AF', 'Sierra Leone': 'AF',
    'Somalia': 'AF', 'South Africa': 'AF', 'South Sudan': 'AF', 'Sudan': 'AF',
    'Tanzania': 'AF', 'Togo': 'AF', 'Tunisia': 'AF', 'Uganda': 'AF', 'Zambia': 'AF',
    'Zimbabwe': 'AF',
    # AFC (Australia moved from OFC to AFC in 2006)
    'Afghanistan': 'AS', 'Australia': 'AS', 'Bahrain': 'AS', 'Bangladesh': 'AS', 'Bhutan': 'AS',
    'Brunei': 'AS', 'Cambodia': 'AS', 'China PR': 'AS', 'East Timor': 'AS', 'Guam': 'AS',
    'Hong Kong': 'AS', 'India': 'AS', 'Indonesia': 'AS', 'Iran': 'AS', 'Iraq': 'AS',
    'Japan': 'AS', 'Jordan': 'AS', 'Kuwait': 'AS', 'Kyrgyzstan': 'AS', 'Laos': 'AS',
    'Lebanon': 'AS', 'Macau': 'AS', 'Malaysia': 'AS', 'Maldives': 'AS', 'Mongolia': 'AS',
    'Myanmar': 'AS', 'Nepal': 'AS', 'North Korea': 'AS', 'Oman': 'AS', 'Pakistan': 'AS',
    'Palestine': 'AS', 'Philippines': 'AS', 'Qatar': 'AS', 'Saudi Arabia': 'AS',
    'Singapore': 'AS', 'South Korea': 'AS', 'Sri Lanka': 'AS', 'Syria': 'AS',
    'Tajikistan': 'AS', 'Thailand': 'AS', 'Turkmenistan': 'AS', 'United Arab Emirates': 'AS',
    'Uzbekistan': 'AS', 'Vietnam': 'AS', 'Yemen': 'AS',
    # OFC
    'American Samoa': 'OC', 'Cook Islands': 'OC', 'Fiji': 'OC', 'New Caledonia': 'OC',
    'New Zealand': 'OC', 'Papua New Guinea': 'OC', 'Samoa': 'OC', 'Solomon Islands': 'OC',
    'Tahiti': 'OC', 'Tonga': 'OC', 'Vanuatu': 'OC',
}


def add_host_continent(matches: pd.DataFrame, results_csv: str = 'results.csv') -> pd.DataFrame:
    """Add (home|away)_in_host_continent flags. Re-reads results.csv for the
    `country` column since the cleaning step dropped it."""
    country_df = pd.read_csv(results_csv, usecols=['date', 'home_team', 'away_team', 'country'])
    country_df['date'] = pd.to_datetime(country_df['date'], format='%Y-%m-%d')
    matches = matches.merge(country_df, on=['date', 'home_team', 'away_team'], how='left')

    host_continent = matches['country'].map(CONTINENT)
    home_continent = matches['home_team'].map(CONTINENT)
    away_continent = matches['away_team'].map(CONTINENT)

    matches['home_in_host_continent'] = (home_continent == host_continent).astype(int)
    matches['away_in_host_continent'] = (away_continent == host_continent).astype(int)
    return matches.drop(columns=['country'])


# === sofifa parsing =======================================================

SOFIFA_TO_CANONICAL = {
    'Czechia': 'Czech Republic',
    'Türkiye': 'Turkey',
    'Cabo Verde': 'Cape Verde',
    'Congo DR': 'DR Congo',
    "Côte d'Ivoire": 'Ivory Coast',
    'Korea Republic': 'South Korea',
    'Curacao': 'Curaçao',
}

_ROW_PATTERN = re.compile(
    r'<a href="https://sofifa\.com/team/\d+/[^/]+/\d+/">([^<]+)</a>'
    r'.*?data-col="oa".*?title="(\d+)"'
    r'.*?data-col="at".*?title="(\d+)"'
    r'.*?data-col="md".*?title="(\d+)"'
    r'.*?data-col="df".*?title="(\d+)"'
    r'.*?data-col="sa"><em>([\d.]+)</em>',
    re.DOTALL,
)


def parse_sofifa_listing(filepath: str) -> pd.DataFrame:
    """Pull team / overall / attack / midfield / defence / starting-XI age
    from a saved sofifa national-teams listing page."""
    with open(filepath, encoding='utf-8') as f:
        html = f.read()
    rows = _ROW_PATTERN.findall(html)
    out = pd.DataFrame(rows, columns=[
        'team', 'overall', 'attack', 'midfield', 'defence', 'starting_xi_avg_age'
    ])
    out['team'] = out['team'].replace(SOFIFA_TO_CANONICAL)
    for c in ['overall', 'attack', 'midfield', 'defence']:
        out[c] = pd.to_numeric(out[c])
    out['starting_xi_avg_age'] = pd.to_numeric(out['starting_xi_avg_age'])
    return out


# === Stefano CSV loader + team_stats combiner =============================

CSV_TO_CANONICAL = {
    'Korea Republic': 'South Korea',
}


def load_teams_hist(filepath: str = 'male_teams_14-24.csv') -> pd.DataFrame:
    """2015-2024 snapshots from Stefano's CSV. league_id 78 = Friendly
    International (the national-team rows). FIFA version N → year 2000+N."""
    teams_hist = pd.read_csv(
        filepath,
        usecols=['team_name', 'fifa_version', 'league_id',
                 'overall', 'attack', 'midfield', 'defence',
                 'starting_xi_average_age'],
    )
    teams_hist = teams_hist[teams_hist['league_id'] == 78].drop(columns=['league_id'])
    teams_hist['team'] = teams_hist['team_name'].replace(CSV_TO_CANONICAL)
    teams_hist['year'] = 2000 + teams_hist['fifa_version'].astype(int)
    teams_hist = teams_hist.rename(columns={'starting_xi_average_age': 'starting_xi_avg_age'})
    return (teams_hist[['team', 'year', 'overall', 'attack', 'midfield', 'defence',
                        'starting_xi_avg_age']]
            .drop_duplicates(subset=['team', 'year'])
            .reset_index(drop=True))


def build_team_stats(teams_hist: pd.DataFrame,
                     fc25: pd.DataFrame,
                     fc26: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, year). FC25 scrape was missing ~30 teams — back-fill
    those with their FC24 row as the best available proxy."""
    fc24 = teams_hist.loc[teams_hist['year'] == 2024,
                          ['team', 'overall', 'attack', 'midfield', 'defence',
                           'starting_xi_avg_age']]
    fc25_filled = pd.concat([
        fc25.assign(year=2025),
        fc24[~fc24['team'].isin(fc25['team'])].assign(year=2025),
    ], ignore_index=True)
    fc26_yr = fc26.assign(year=2026)

    team_stats = pd.concat([teams_hist, fc25_filled, fc26_yr], ignore_index=True)
    return (team_stats[['team', 'year', 'overall', 'attack', 'midfield',
                        'defence', 'starting_xi_avg_age']]
            .sort_values(['team', 'year'])
            .reset_index(drop=True))


# === merge_asof team_stats onto matches ===================================

# 2015-2024: `update_as_of` snapshot dates from the Stefano CSV.
# FC25 / FC26: public release dates (CSV doesn't cover these yet).
SNAPSHOT_DATE = {
    2015: '2014-09-18', 2016: '2015-09-21', 2017: '2016-09-20',
    2018: '2017-09-18', 2019: '2018-08-21', 2020: '2019-09-19',
    2021: '2020-09-23', 2022: '2021-09-23', 2023: '2022-09-26',
    2024: '2023-09-22', 2025: '2024-09-27', 2026: '2025-09-26',
}


def attach_team_stats(matches: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Time-aware join: each match picks the most recent sofifa snapshot
    available at kickoff (per team). Adds 5 home_* and 5 away_* columns."""
    ts = team_stats.copy()
    ts['snapshot_date'] = pd.to_datetime(ts['year'].map(SNAPSHOT_DATE))
    ts = ts.drop(columns=['year']).sort_values('snapshot_date').reset_index(drop=True)

    home_stats = ts.rename(columns={
        'team': 'home_team',
        'overall': 'home_overall', 'attack': 'home_attack',
        'midfield': 'home_midfield', 'defence': 'home_defence',
        'starting_xi_avg_age': 'home_starting_xi_avg_age',
    })
    away_stats = ts.rename(columns={
        'team': 'away_team',
        'overall': 'away_overall', 'attack': 'away_attack',
        'midfield': 'away_midfield', 'defence': 'away_defence',
        'starting_xi_avg_age': 'away_starting_xi_avg_age',
    })

    out = pd.merge_asof(
        matches.sort_values('date').reset_index(drop=True),
        home_stats, left_on='date', right_on='snapshot_date',
        by='home_team', direction='backward',
    ).drop(columns=['snapshot_date'])

    out = pd.merge_asof(
        out.sort_values('date').reset_index(drop=True),
        away_stats, left_on='date', right_on='snapshot_date',
        by='away_team', direction='backward',
    ).drop(columns=['snapshot_date'])

    return out


# === pipeline =============================================================

def main():
    full = pd.read_csv('results.csv')
    full['date'] = pd.to_datetime(full['date'], format='%Y-%m-%d')
    full['home_score'] = pd.to_numeric(full['home_score'], errors='coerce').astype('Int64')
    full['away_score'] = pd.to_numeric(full['away_score'], errors='coerce').astype('Int64')

    full_with_elo, final_ratings = compute_elo_history(full)
    df = full_with_elo[full_with_elo['date'] > pd.Timestamp('2020-12-31')].copy()

    df = compute_recent_goals(df)
    df = compute_recent_wdl(df)
    df = compute_rest_days(df)
    df = add_host_continent(df)

    teams_hist = load_teams_hist()
    fc25 = parse_sofifa_listing('sofifa_teams_fc25.html')
    fc26 = parse_sofifa_listing('sofifa_teams_fc26.html')
    team_stats = build_team_stats(teams_hist, fc25, fc26)

    df_with_stats = attach_team_stats(df, team_stats)
    return df_with_stats, final_ratings


if __name__ == '__main__':
    df, ratings = main()
    print(df.tail())
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:15]
    for team, r in top:
        print(f"{team:25s} {r:7.1f}")
