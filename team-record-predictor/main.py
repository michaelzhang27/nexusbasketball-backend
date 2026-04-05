"""
Conference win probability predictor.

Given a team name and a list of that team's player BPM scores, finds every
other team in the same conference (from the 2026 teams dataset), pulls each
opponent's players from the player stats CSV, and runs a minutes-weighted
win probability simulation for every conference matchup.

Usage
-----
    result = conference_win_probabilities(
        team_name="Duke",
        team_bpm=[16.62, 11.13, 10.77, 11.34, 10.07, 8.76, 9.77, 7.63, 6.30]
    )
    # result -> {"Virginia": 0.87, "North Carolina": 0.74, ...}
"""

import math
import os
import random
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

# ── File paths (relative to this file) ──────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
TEAMS_CSV   = os.path.join(_DIR, "2026 teams data set.csv")
PLAYERS_CSV = os.path.join(_DIR, "mbb_player_team_stats_200min_2026 copy.csv")

# ── Calibrated model parameters (adjusted BPM, k recalibrated to 0.50) ──────
K = 0.50
LEAGUE_MEAN = 6.5226

# ── Mascot list for stripping school names ───────────────────────────────────
_MASCOTS = [
    'Blue Devils','Wildcats','Bulldogs','Tigers','Eagles','Bears','Wolves',
    'Spartans','Wolverines','Trojans','Gators','Cougars','Aggies','Cardinals',
    'Huskies','Cavaliers','Commodores','Cornhuskers','Razorbacks','Volunteers',
    'Cyclones','Jayhawks','Boilermakers','Tar Heels','Badgers','Buckeyes',
    'Bruins','Longhorns','Crimson Tide','Fighting Illini','Nittany Lions',
    'Orange','Panthers','Mountaineers','Seminoles','Yellow Jackets','Hokies',
    'Terrapins','Demon Deacons','Golden Gophers','Red Raiders','Pirates',
    'Bearcats','Owls','Mustangs','Horned Frogs','Rainbow Warriors','Gaels',
    'Friars','Scarlet Knights','Sun Devils','Wolf Pack','Rebels','Mean Green',
    'Roadrunners','Miners','Lobos','Thunderbirds','Flyers','Rockets','Zips',
    'RedHawks','Redhawks','Billikens','Bonnies','Explorers','Hawks','Ramblers',
    'Dons','Musketeers','Hoyas','Lions','Big Red','Quakers','Leopards','Camels',
    'Crusaders','Flying Dutchmen','Pride','Seawolves','Retrievers','Terriers',
    'Peacocks','Colonials','Patriots','Flames','Monarchs','Tribe','Colonels',
    'Mocs','Buccaneers','Chanticleers','Catamounts','Keydets','Paladins',
    'Spiders','Greyhounds','Mastodons','Penguins','Golden Eagles','Phoenix',
    'Braves','Utes','Coyotes','Beavers','Ducks','Bison','Jackrabbits',
    'Lumberjacks','Norse','Racers','Jaguars','Royals','Dukes','Flashes',
    'Falcons','Leathernecks','Delta Devils','Blue Raiders','Mavericks',
    'Thundering Herd','Bobcats','Anteaters','Matadors','Gauchos','Tritons',
    'River Hawks','Minutemen','Gorillas','Ospreys','Dolphins','Lancers',
    'Storm','Saints','Pilots','Toreros','Waves','Pioneers','Stags','Rams',
    'Sooners','Cowboys','Golden Bears','Grizzlies','Hawkeyes','Cardinal',
    'Hatters','Green Wave','Golden Hurricane','Griffins','Penmen','Great Danes',
    'Fighting Hawks','Warhawks','Blazers','Vols','Roos','Lopes','Bearkats',
    'Revolutionaries','Vikings','Beacons','Broncos','Shockers','Buffaloes',
    'Redbirds','Raiders','Vandals','Bluejays','Chargers','Jaspers','Knights',
    'Lakers','Titans','Warriors','Chippewas','Islanders','Dragons','Blue Hens',
    'Crimson','Privateers','Texans','Governors','Seahawks','Bisons','49ers',
    'Beach','Hornets','Beach',
]

# Manual overrides: teams dataset name -> ESPN display name in players CSV
_MANUAL = {
    'N.C. State':           'NC State Wolfpack',
    'Miami FL':             'Miami Hurricanes',
    'Connecticut':          'UConn Huskies',
    "St. John's":           "St. John's Red Storm",
    'Florida St.':          'Florida State Seminoles',
    'Iowa St.':             'Iowa State Cyclones',
    'Ohio St.':             'Ohio State Buckeyes',
    'Penn St.':             'Penn State Nittany Lions',
    'Utah St.':             'Utah State Aggies',
    'McNeese St.':          'McNeese Cowboys',
    'Ball St.':             'Ball State Cardinals',
    'Kent St.':             'Kent State Golden Flashes',
    'Indiana St.':          'Indiana State Sycamores',
    'Idaho St.':            'Idaho State Bengals',
    'IU Indy':              'IU Indianapolis Jaguars',
    'Appalachian St.':      'App State Mountaineers',
    'Mississippi':          'Ole Miss Rebels',
    'NJIT':                 'NJIT Highlanders',
    'FIU':                  'Florida International Panthers',
    'DePaul':               'DePaul Blue Demons',
    'Navy':                 'Navy Midshipmen',
    'Army':                 'Army Black Knights',
    'UCF':                  'UCF Knights',
    'Illinois Chicago':     'UIC Flames',
    'Oakland':              'Oakland Golden Grizzlies',
    'Notre Dame':           'Notre Dame Fighting Irish',
    'Indiana':              'Indiana Hoosiers',
    'Penn':                 'Pennsylvania Quakers',
    'Harvard':              'Harvard Crimson',
    'Dartmouth':            'Dartmouth Big Green',
    'American':             'American University Eagles',
    'Maine':                'Maine Black Bears',
    'Delaware':             'Delaware Blue Hens',
    'Delaware St.':         'Delaware State Hornets',
    'Drexel':               'Drexel Dragons',
    'UC Riverside':         'UC Riverside Highlanders',
    'Nebraska Omaha':       'Omaha Mavericks',
    'UMKC':                 'Kansas City Roos',
    'UNC Wilmington':       'UNC Wilmington Seahawks',
    'Cal Baptist':          'California Baptist Lancers',
    'Cal St. Fullerton':    'Cal State Fullerton Titans',
    'Central Michigan':     'Central Michigan Chippewas',
    'Southern Indiana':     'Southern Indiana Screaming Eagles',
    'San Jose St.':         'San José State Spartans',
    'Tennessee Martin':     'UT Martin Skyhawks',
    'Iowa St.':             'Iowa State Cyclones',
    'Gardner Webb':         "Gardner-Webb Runnin' Bulldogs",
    'Loyola MD':            'Loyola Maryland Greyhounds',
    'Charlotte':            'Charlotte 49ers',
    'Texas A&M Corpus Chris': 'Texas A&M-Corpus Christi Islanders',
    'Sacramento St.':       'Sacramento State Hornets',
    'Long Beach St.':       'Long Beach State Beach',
    'Louisiana Monroe':     'UL Monroe Warhawks',
    'Southeastern Louisiana': 'SE Louisiana Lions',
    'Marist':               'Marist Red Foxes',
    'Manhattan':            'Manhattan Jaspers',
    'Canisius':             'Canisius Golden Griffins',
    'Niagara':              'Niagara Purple Eagles',
    'LIU':                  'LIU Sharks',
    'Wagner':               'Wagner Seahawks',
    'Creighton':            'Creighton Bluejays',
    'Merrimack':            'Merrimack Warriors',
    'New Orleans':          'New Orleans Privateers',
    'Tarleton St.':         'Tarleton State Texans',
    'USC Upstate':          'South Carolina Upstate Spartans',
    'Stonehill':            'Stonehill Skyhawks',
    'Mercyhurst':           'Mercyhurst Lakers',
    'Bellarmine':           'Bellarmine Knights',
    'Austin Peay':          'Austin Peay Governors',
    'Presbyterian':         'Presbyterian Blue Hose',
    'Rider':                'Rider Broncs',
    'Radford':              'Radford Highlanders',
    'Evansville':           'Evansville Purple Aces',
    'Florida A&M':          'Florida A&M Rattlers',
    'Alabama St.':          'Alabama State Hornets',
    'Detroit Mercy':        'Detroit Mercy Titans',
    'St. Thomas':           'Saint Thomas Tommies',
    'Idaho':                'Idaho Vandals',
    'Queens':               'Queens University Royals',
    'New Haven':            'New Haven Chargers',
    'Utah Tech':            'Utah Tech Trailblazers',
    'Lipscomb':             'Lipscomb Bisons',
    'Buffalo':              'Buffalo Bulls',
    'Campbell':             'Campbell Fighting Camels',
    'Lehigh':               'Lehigh Mountain Hawks',
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_mascot(name: str) -> str:
    for m in sorted(_MASCOTS, key=len, reverse=True):
        if name.endswith(' ' + m):
            return name[:-(len(m) + 1)].strip()
    return name


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _weighted_rating(bpm: list[float], minutes: list[float]) -> float:
    total_min = sum(minutes)
    if total_min <= 0:
        return sum(bpm) / len(bpm)
    return sum(b * m for b, m in zip(bpm, minutes)) / total_min


def _win_prob(rating_a: float, rating_b: float) -> float:
    return _sigmoid(K * (rating_a - rating_b))


def _load_data():
    teams_df   = pd.read_csv(TEAMS_CSV)
    players_df = pd.read_csv(PLAYERS_CSV)
    players_df = players_df.dropna(subset=['box_plus_minus', 'general_minutes'])

    # Build stripped-name lookup for player teams
    player_team_names = (
        players_df[['team_id', 'team_display_name']]
        .drop_duplicates()
        .copy()
    )
    player_team_names['stripped'] = player_team_names['team_display_name'].apply(_strip_mascot)

    return teams_df, players_df, player_team_names


def _match_team_name(dataset_name: str, player_team_names: pd.DataFrame) -> str | None:
    """Map a team name from the teams dataset to an ESPN display name in the players CSV."""
    # Check manual overrides first
    if dataset_name in _MANUAL:
        return _MANUAL[dataset_name]

    # Fuzzy match against stripped player team names
    best_score, best_name = 0.0, None
    for _, row in player_team_names.iterrows():
        s = SequenceMatcher(None, dataset_name.lower(), row['stripped'].lower()).ratio()
        if s > best_score:
            best_score, best_name = s, row['team_display_name']

    return best_name if best_score >= 0.80 else None


def _get_team_players(display_name: str, players_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows from players_df matching the given ESPN display name."""
    return players_df[players_df['team_display_name'] == display_name]


# ── Main helper ──────────────────────────────────────────────────────────────

def conference_win_probabilities(
    team_name: str,
    team_bpm: list[float],
    team_minutes: list[float] | None = None,
    simulations: int = 10_000,
) -> dict:
    """
    Predict win probability for a team against every opponent in its conference,
    then run a Monte Carlo season simulation (each opponent played twice).

    Parameters
    ----------
    team_name : str
        Team name as it appears in the 2026 teams dataset (e.g. "Duke", "Iowa St.").
    team_bpm : list of float
        BPM scores for each player on the input team (use adjusted_bpm from
        player_bpm_adjusted_2026.csv for best accuracy).
    team_minutes : list of float or None
        Minutes for each player in team_bpm. If provided, uses minutes-weighted
        ratings; otherwise uses plain average.
    simulations : int
        Number of Monte Carlo season simulations to run (default 10,000).

    Returns
    -------
    dict with two keys:
        "win_probabilities" : dict mapping opponent name -> win probability (0–1),
                              sorted descending by win probability.
        "monte_carlo"       : dict with:
            "distribution"      – {wins: count} for all observed season win totals,
                                  sorted ascending by win total.
            "most_likely_wins"  – the win total that occurred most often.
            "simulations"       – number of simulations run.
            "max_possible_wins" – 2 × number of matched opponents.
    """
    teams_df, players_df, player_team_names = _load_data()

    # ── Find the input team's conference ──
    row = teams_df[teams_df['TEAM'].str.lower() == team_name.lower()]
    if row.empty:
        # Try fuzzy match
        best_score, best_row = 0.0, None
        for _, r in teams_df.iterrows():
            s = SequenceMatcher(None, team_name.lower(), r['TEAM'].lower()).ratio()
            if s > best_score:
                best_score, best_row = s, r
        if best_score < 0.6:
            raise ValueError(f"Team '{team_name}' not found in the 2026 teams dataset.")
        row = best_row.to_frame().T
        print(f"  Matched '{team_name}' to '{row.iloc[0]['TEAM']}'")

    conference = row.iloc[0]['CONF']
    matched_team_name = row.iloc[0]['TEAM']
    print(f"  Team: {matched_team_name} | Conference: {conference}")

    # ── Compute input team's rating ──
    if team_minutes is not None and len(team_minutes) == len(team_bpm):
        team_rating = _weighted_rating(team_bpm, team_minutes)
    else:
        team_rating = sum(team_bpm) / len(team_bpm)

    # ── Get all conference opponents ──
    conf_teams = teams_df[
        (teams_df['CONF'] == conference) &
        (teams_df['TEAM'].str.lower() != matched_team_name.lower())
    ]['TEAM'].tolist()

    print(f"  {len(conf_teams)} conference opponents found.")

    # ── Compute win probability vs each opponent ──
    results = {}
    unmatched = []

    for opp_dataset_name in conf_teams:
        opp_display = _match_team_name(opp_dataset_name, player_team_names)
        if opp_display is None:
            unmatched.append(opp_dataset_name)
            continue

        opp_players = _get_team_players(opp_display, players_df)
        if opp_players.empty:
            unmatched.append(opp_dataset_name)
            continue

        opp_bpm     = opp_players['box_plus_minus'].tolist()
        opp_minutes = opp_players['general_minutes'].tolist()
        opp_rating  = _weighted_rating(opp_bpm, opp_minutes)

        prob = _win_prob(team_rating, opp_rating)
        results[opp_dataset_name] = round(prob, 4)

    if unmatched:
        print(f"  Could not match {len(unmatched)} opponent(s): {unmatched}")

    # Sort descending by win probability
    sorted_probs = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # ── Monte Carlo season simulation ────────────────────────────────────────
    # Each opponent is played twice; each game is an independent Bernoulli trial.
    prob_list = list(results.values())
    win_counts: Counter = Counter()

    for _ in range(simulations):
        season_wins = sum(
            (1 if random.random() < p else 0) +
            (1 if random.random() < p else 0)
            for p in prob_list
        )
        win_counts[season_wins] += 1

    most_likely_wins = win_counts.most_common(1)[0][0]
    distribution = dict(sorted(win_counts.items()))

    return {
        "win_probabilities": sorted_probs,
        "monte_carlo": {
            "distribution": distribution,
            "most_likely_wins": most_likely_wins,
            "simulations": simulations,
            "max_possible_wins": 2 * len(results),
        },
    }


def print_conference_probabilities(
    team_name: str,
    team_bpm: list[float],
    team_minutes: list[float] | None = None,
    simulations: int = 10_000,
) -> None:
    """Pretty-print the conference win probability table and Monte Carlo results."""
    print(f"\nConference win probabilities for: {team_name}")
    print("=" * 50)

    output = conference_win_probabilities(team_name, team_bpm, team_minutes, simulations)
    probs = output["win_probabilities"]
    mc    = output["monte_carlo"]

    print(f"\n{'Opponent':<30} {'Win Prob':>10}  {'Loss Prob':>10}")
    print("-" * 54)
    for opp, prob in probs.items():
        print(f"{opp:<30} {prob*100:>9.1f}%  {(1-prob)*100:>9.1f}%")

    print(f"\n── Monte Carlo Season Simulation ({mc['simulations']:,} runs, each opponent played twice) ──")
    print(f"{'Wins':<8} {'Times':>8}  {'Share':>8}")
    print("-" * 28)
    for wins, count in mc["distribution"].items():
        share = count / mc["simulations"] * 100
        print(f"{wins:<8} {count:>8,}  {share:>7.1f}%")
    print(f"\nMost likely record: {mc['most_likely_wins']}–{mc['max_possible_wins'] - mc['most_likely_wins']} "
          f"(out of {mc['max_possible_wins']} conference games)")
    print()


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Duke's adjusted BPM scores from player_bpm_adjusted_2026.csv
    duke_bpm = [16.62, 11.13, 10.77, 11.34, 10.07, 8.76, 9.77, 7.63, 6.30]

    print_conference_probabilities("Duke", duke_bpm)
