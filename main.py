"""
Nexus Analytics — FastAPI Backend
Reads player data from local CSV files and serves it via REST endpoints.
Post-beta seam: replace CSV reads with DB queries at the data-loading functions.

Run with:
    uvicorn main:app --reload --port 8000
"""

import csv
import os
import sys
import math
import importlib.util
from pathlib import Path
from typing import Optional, List, Any
from fastapi import FastAPI, Query, Depends, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from db import (
    get_current_user_id, fetch_user_data,
    upsert_scenario, remove_scenario,
    upsert_player_notes,
    upsert_model, remove_model,
    create_confirmed_user,
    get_supabase,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
# CSVs now live alongside the backend in backend/players/
BASE_DIR = Path(__file__).parent / "players"
ALL_PLAYERS_CSV        = BASE_DIR / "all_players.csv"
TRANSFERS_CSV          = BASE_DIR / "transfers.csv"
WOMENS_ALL_PLAYERS_CSV = BASE_DIR / "womens_all_players.csv"
WOMENS_TRANSFERS_CSV   = BASE_DIR / "womens_transfers.csv"
PREDICTOR_DIR        = Path(__file__).parent / "player-stat-predictor"
TEAM_PREDICTOR_DIR   = Path(__file__).parent / "team-record-predictor"

# ── Predictor — loaded once at startup ────────────────────────────────────────
_predictor = None
_team_predictor = None

def _load_predictor():
    global _predictor
    entry = PREDICTOR_DIR / "main.py"
    # Add predictor directory to sys.path so `from predictor import ...` resolves
    pred_dir = str(PREDICTOR_DIR)
    if pred_dir not in sys.path:
        sys.path.insert(0, pred_dir)
    spec = importlib.util.spec_from_file_location("player_stat_predictor", entry)
    mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    _predictor = mod


def _load_team_predictor():
    global _team_predictor
    entry = TEAM_PREDICTOR_DIR / "main.py"
    team_dir = str(TEAM_PREDICTOR_DIR)
    if team_dir not in sys.path:
        sys.path.insert(0, team_dir)
    spec = importlib.util.spec_from_file_location("team_record_predictor", entry)
    mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    _team_predictor = mod

# Raw feature vectors for each player (populated by load_players at startup)
_raw_features_cache: dict[str, dict] = {}

# 29 features the XGBoost models expect, in order
PREDICTOR_FEATURES = [
    "weight", "height", "position_id", "experience_years",
    "defensive_avg_defensive_rebounds", "defensive_avg_blocks", "defensive_avg_steals",
    "general_avg_minutes", "general_avg_rebounds", "general_avg_fouls",
    "offensive_field_goal_pct", "offensive_free_throws",
    "offensive_avg_field_goals_made", "offensive_avg_field_goals_attempted",
    "offensive_avg_three_point_field_goals_made", "offensive_avg_three_point_field_goals_attempted",
    "offensive_avg_free_throws_made", "offensive_avg_free_throws_attempted",
    "offensive_avg_points", "offensive_avg_offensive_rebounds", "offensive_avg_assists",
    "offensive_avg_turnovers", "offensive_three_point_field_goal_pct",
    "offensive_avg_two_point_field_goals_made", "offensive_avg_two_point_field_goals_attempted",
    "offensive_two_point_field_goal_pct", "offensive_shooting_efficiency",
    "offensive_scoring_efficiency", "general_minutes",
]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nexus Analytics API", version="0.1.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://www.nexusbasketball.org", "https://nexusbasketball.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ─────────────────────────────────────────────────────────────────────
class PlayerStats(BaseModel):
    season: Optional[str] = None
    ppg: float
    rpg: float
    apg: float
    spg: float
    bpg: float
    topg: float
    fgPct: float   # 0–100
    fg3Pct: float  # 0–100
    ftPct: float   # 0–100
    efgPct: float  # 0–100
    tsPct: float   # 0–100
    usagePct: float
    ortg: float
    drtg: float
    bpm: float
    winShares: float
    per: float
    minutesPerGame: float
    # Per-game counting stats for per-40 fit score computation
    fgaPerGame: float = 0.0
    ftaPerGame: float = 0.0
    fg3mPerGame: float = 0.0
    orebPerGame: float = 0.0
    drebPerGame: float = 0.0
    foulsPerGame: float = 0.0

class Player(PlayerStats):
    id: str
    name: str
    position: str
    previousSchool: str
    conference: str
    classYear: str
    eligibilityRemaining: int
    height: str
    weight: int
    hometown: str
    portalEntryDate: str
    portalStatus: str
    avatarColor: str
    nilEstimate: list[int]
    notes: list[str]
    isReturner: bool
    photoUrl: Optional[str] = None


class SearchResult(BaseModel):
    total: int
    offset: int
    limit: int
    players: list[Player]

class PredictionResult(BaseModel):
    player_id: str
    projected_mpg: float
    # Scoring
    points: float
    fg_made: float
    fg_attempted: float
    fg_pct: float            # decimal 0–1
    fg3_made: float
    fg3_attempted: float
    fg3_pct: float           # decimal 0–1
    fg2_made: float
    fg2_attempted: float
    fg2_pct: float           # decimal 0–1
    ft_made: float
    ft_attempted: float
    ft_pct: float            # decimal 0–1
    # Playmaking
    assists: float
    turnovers: float
    # Rebounding
    offensive_rebounds: float
    defensive_rebounds: float
    total_rebounds: float
    # Defense
    steals: float
    blocks: float
    fouls: float
    # Efficiency (ESPN proprietary scalars — values near 1.0)
    shooting_efficiency: float
    scoring_efficiency: float

class BatchPredictionItem(BaseModel):
    player_id: str
    projected_mpg: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    players: list[BatchPredictionItem]

# ── Conference mapping ─────────────────────────────────────────────────────────
# 2025-26 D1 conference alignments
CONFERENCE_MAP: dict[str, str] = {
    # ACC
    "Boston College Eagles": "ACC",
    "Clemson Tigers": "ACC",
    "Duke Blue Devils": "ACC",
    "Florida State Seminoles": "ACC",
    "Georgia Tech Yellow Jackets": "ACC",
    "Louisville Cardinals": "ACC",
    "Miami Hurricanes": "ACC",
    "NC State Wolfpack": "ACC",
    "North Carolina Tar Heels": "ACC",
    "Notre Dame Fighting Irish": "ACC",
    "Pittsburgh Panthers": "ACC",
    "SMU Mustangs": "ACC",
    "Stanford Cardinal": "ACC",
    "Syracuse Orange": "ACC",
    "Virginia Cavaliers": "ACC",
    "Virginia Tech Hokies": "ACC",
    "Wake Forest Demon Deacons": "ACC",
    "California Golden Bears": "ACC",
    # Big 12
    "Arizona Wildcats": "Big 12",
    "Arizona State Sun Devils": "Big 12",
    "BYU Cougars": "Big 12",
    "Baylor Bears": "Big 12",
    "Cincinnati Bearcats": "Big 12",
    "Colorado Buffaloes": "Big 12",
    "Houston Cougars": "Big 12",
    "Iowa State Cyclones": "Big 12",
    "Kansas Jayhawks": "Big 12",
    "Kansas State Wildcats": "Big 12",
    "Oklahoma State Cowboys": "Big 12",
    "TCU Horned Frogs": "Big 12",
    "Texas Longhorns": "Big 12",
    "Texas Tech Red Raiders": "Big 12",
    "UCF Knights": "Big 12",
    "Utah Utes": "Big 12",
    "West Virginia Mountaineers": "Big 12",
    # Big Ten
    "Illinois Fighting Illini": "Big Ten",
    "Indiana Hoosiers": "Big Ten",
    "Iowa Hawkeyes": "Big Ten",
    "Maryland Terrapins": "Big Ten",
    "Michigan Wolverines": "Big Ten",
    "Michigan State Spartans": "Big Ten",
    "Minnesota Golden Gophers": "Big Ten",
    "Nebraska Cornhuskers": "Big Ten",
    "Northwestern Wildcats": "Big Ten",
    "Ohio State Buckeyes": "Big Ten",
    "Oregon Ducks": "Big Ten",
    "Penn State Nittany Lions": "Big Ten",
    "Purdue Boilermakers": "Big Ten",
    "Rutgers Scarlet Knights": "Big Ten",
    "UCLA Bruins": "Big Ten",
    "USC Trojans": "Big Ten",
    "Washington Huskies": "Big Ten",
    "Wisconsin Badgers": "Big Ten",
    # Big East
    "Butler Bulldogs": "Big East",
    "Creighton Bluejays": "Big East",
    "DePaul Blue Demons": "Big East",
    "Georgetown Hoyas": "Big East",
    "Marquette Golden Eagles": "Big East",
    "Providence Friars": "Big East",
    "Seton Hall Pirates": "Big East",
    "St. John's Red Storm": "Big East",
    "UConn Huskies": "Big East",
    "Villanova Wildcats": "Big East",
    "Xavier Musketeers": "Big East",
    # SEC
    "Alabama Crimson Tide": "SEC",
    "Arkansas Razorbacks": "SEC",
    "Auburn Tigers": "SEC",
    "Florida Gators": "SEC",
    "Georgia Bulldogs": "SEC",
    "Kentucky Wildcats": "SEC",
    "LSU Tigers": "SEC",
    "Mississippi State Bulldogs": "SEC",
    "Missouri Tigers": "SEC",
    "Ole Miss Rebels": "SEC",
    "Oklahoma Sooners": "SEC",
    "South Carolina Gamecocks": "SEC",
    "Tennessee Volunteers": "SEC",
    "Texas A&M Aggies": "SEC",
    "Vanderbilt Commodores": "SEC",
    # Mountain West
    "Air Force Falcons": "Mountain West",
    "Boise State Broncos": "Mountain West",
    "Colorado State Rams": "Mountain West",
    "Fresno State Bulldogs": "Mountain West",
    "Hawai'i Rainbow Warriors": "Mountain West",
    "New Mexico Lobos": "Mountain West",
    "Nevada Wolf Pack": "Mountain West",
    "San Diego State Aztecs": "Mountain West",
    "San José State Spartans": "Mountain West",
    "UNLV Rebels": "Mountain West",
    "Utah State Aggies": "Mountain West",
    "Wyoming Cowboys": "Mountain West",
    # Atlantic 10
    "Davidson Wildcats": "Atlantic 10",
    "Dayton Flyers": "Atlantic 10",
    "Duquesne Dukes": "Atlantic 10",
    "Fordham Rams": "Atlantic 10",
    "George Mason Patriots": "Atlantic 10",
    "George Washington Revolutionaries": "Atlantic 10",
    "La Salle Explorers": "Atlantic 10",
    "Loyola Chicago Ramblers": "Atlantic 10",
    "Massachusetts Minutemen": "Atlantic 10",
    "Rhode Island Rams": "Atlantic 10",
    "Richmond Spiders": "Atlantic 10",
    "Saint Joseph's Hawks": "Atlantic 10",
    "Saint Louis Billikens": "Atlantic 10",
    "Saint Peter's Peacocks": "Atlantic 10",
    "VCU Rams": "Atlantic 10",
    # American Athletic Conference
    "Charlotte 49ers": "AAC",
    "East Carolina Pirates": "AAC",
    "Florida Atlantic Owls": "AAC",
    "Memphis Tigers": "AAC",
    "North Texas Mean Green": "AAC",
    "Rice Owls": "AAC",
    "South Florida Bulls": "AAC",
    "Temple Owls": "AAC",
    "Tulane Green Wave": "AAC",
    "Tulsa Golden Hurricane": "AAC",
    "UAB Blazers": "AAC",
    "Wichita State Shockers": "AAC",
    "UTSA Roadrunners": "AAC",
    # Conference USA
    "Florida International Panthers": "CUSA",
    "Louisiana Tech Bulldogs": "CUSA",
    "Middle Tennessee Blue Raiders": "CUSA",
    "New Mexico State Aggies": "CUSA",
    "Old Dominion Monarchs": "CUSA",
    "Sam Houston Bearkats": "CUSA",
    "Western Kentucky Hilltoppers": "CUSA",
    "UTEP Miners": "CUSA",
    "Jacksonville State Gamecocks": "CUSA",
    # Sun Belt
    "App State Mountaineers": "Sun Belt",
    "Arkansas State Red Wolves": "Sun Belt",
    "Coastal Carolina Chanticleers": "Sun Belt",
    "East Texas A&M Lions": "Sun Belt",
    "Georgia Southern Eagles": "Sun Belt",
    "Georgia State Panthers": "Sun Belt",
    "James Madison Dukes": "Sun Belt",
    "Louisiana Ragin' Cajuns": "Sun Belt",
    "Marshall Thundering Herd": "Sun Belt",
    "South Alabama Jaguars": "Sun Belt",
    "Southern Miss Golden Eagles": "Sun Belt",
    "Texas State Bobcats": "Sun Belt",
    "Troy Trojans": "Sun Belt",
    "UL Monroe Warhawks": "Sun Belt",
    # Missouri Valley
    "Bradley Braves": "Missouri Valley",
    "Drake Bulldogs": "Missouri Valley",
    "Evansville Purple Aces": "Missouri Valley",
    "Illinois State Redbirds": "Missouri Valley",
    "Indiana State Sycamores": "Missouri Valley",
    "Missouri State Bears": "Missouri Valley",
    "Northern Iowa Panthers": "Missouri Valley",
    "Oral Roberts Golden Eagles": "Missouri Valley",
    "Southern Illinois Salukis": "Missouri Valley",
    "Valparaiso Beacons": "Missouri Valley",
    "Belmont Bruins": "Missouri Valley",
    "Murray State Racers": "Missouri Valley",
    "UT Martin Skyhawks": "Missouri Valley",
    # WCC
    "Gonzaga Bulldogs": "WCC",
    "Loyola Marymount Lions": "WCC",
    "Pacific Tigers": "WCC",
    "Pepperdine Waves": "WCC",
    "Portland Pilots": "WCC",
    "Saint Mary's Gaels": "WCC",
    "San Diego Toreros": "WCC",
    "San Francisco Dons": "WCC",
    "Santa Clara Broncos": "WCC",
    "Seattle U Redhawks": "WCC",
    # Ivy League
    "Brown Bears": "Ivy League",
    "Columbia Lions": "Ivy League",
    "Cornell Big Red": "Ivy League",
    "Dartmouth Big Green": "Ivy League",
    "Harvard Crimson": "Ivy League",
    "Pennsylvania Quakers": "Ivy League",
    "Princeton Tigers": "Ivy League",
    "Yale Bulldogs": "Ivy League",
    # Patriot League
    "American University Eagles": "Patriot League",
    "Army Black Knights": "Patriot League",
    "Bucknell Bison": "Patriot League",
    "Colgate Raiders": "Patriot League",
    "Holy Cross Crusaders": "Patriot League",
    "Lafayette Leopards": "Patriot League",
    "Lehigh Mountain Hawks": "Patriot League",
    "Loyola Maryland Greyhounds": "Patriot League",
    "Navy Midshipmen": "Patriot League",
    # Southern Conference
    "The Citadel Bulldogs": "Southern",
    "East Tennessee State Buccaneers": "Southern",
    "Furman Paladins": "Southern",
    "Mercer Bears": "Southern",
    "Samford Bulldogs": "Southern",
    "VMI Keydets": "Southern",
    "Western Carolina Catamounts": "Southern",
    "Wofford Terriers": "Southern",
    "Chattanooga Mocs": "Southern",
    # MAAC
    "Canisius Golden Griffins": "MAAC",
    "Fairfield Stags": "MAAC",
    "Iona Gaels": "MAAC",
    "Manhattan Jaspers": "MAAC",
    "Marist Red Foxes": "MAAC",
    "Monmouth Hawks": "MAAC",
    "Niagara Purple Eagles": "MAAC",
    "Quinnipiac Bobcats": "MAAC",
    "Rider Broncs": "MAAC",
    "Siena Saints": "MAAC",
    # MAC
    "Akron Zips": "MAC",
    "Ball State Cardinals": "MAC",
    "Bowling Green Falcons": "MAC",
    "Buffalo Bulls": "MAC",
    "Central Michigan Chippewas": "MAC",
    "Eastern Michigan Eagles": "MAC",
    "Kent State Golden Flashes": "MAC",
    "Miami (OH) RedHawks": "MAC",
    "Northern Illinois Huskies": "MAC",
    "Ohio Bobcats": "MAC",
    "Toledo Rockets": "MAC",
    "Western Michigan Broncos": "MAC",
    # Horizon League
    "Cleveland State Vikings": "Horizon",
    "Detroit Mercy Titans": "Horizon",
    "Green Bay Phoenix": "Horizon",
    "IU Indianapolis Jaguars": "Horizon",
    "Milwaukee Panthers": "Horizon",
    "Northern Kentucky Norse": "Horizon",
    "Oakland Golden Grizzlies": "Horizon",
    "Purdue Fort Wayne Mastodons": "Horizon",
    "Robert Morris Colonials": "Horizon",
    "Wright State Raiders": "Horizon",
    "Youngstown State Penguins": "Horizon",
    # CAA
    "Campbell Fighting Camels": "CAA",
    "Charleston Cougars": "CAA",
    "Delaware Blue Hens": "CAA",
    "Drexel Dragons": "CAA",
    "Elon Phoenix": "CAA",
    "Hampton Pirates": "CAA",
    "Hofstra Pride": "CAA",
    "NJIT Highlanders": "CAA",
    "Northeastern Huskies": "CAA",
    "Stony Brook Seawolves": "CAA",
    "Towson Tigers": "CAA",
    "UNC Wilmington Seahawks": "CAA",
    "William & Mary Tribe": "CAA",
    # Big South
    "Charleston Southern Buccaneers": "Big South",
    "Gardner-Webb Runnin' Bulldogs": "Big South",
    "High Point Panthers": "Big South",
    "Longwood Lancers": "Big South",
    "Presbyterian Blue Hose": "Big South",
    "Radford Highlanders": "Big South",
    "Winthrop Eagles": "Big South",
    "UNC Asheville Bulldogs": "Big South",
    # Ohio Valley
    "Austin Peay Governors": "OVC",
    "Eastern Illinois Panthers": "OVC",
    "Eastern Kentucky Colonels": "OVC",
    "Morehead State Eagles": "OVC",
    "Tennessee State Tigers": "OVC",
    "Tennessee Tech Golden Eagles": "OVC",
    # Southland
    "Houston Christian Huskies": "Southland",
    "Incarnate Word Cardinals": "Southland",
    "Lamar Cardinals": "Southland",
    "McNeese Cowboys": "Southland",
    "New Orleans Privateers": "Southland",
    "Nicholls Colonels": "Southland",
    "Northwestern State Demons": "Southland",
    "SE Louisiana Lions": "Southland",
    "Texas A&M-Corpus Christi Islanders": "Southland",
    "UT Rio Grande Valley Vaqueros": "Southland",
    "Stephen F. Austin Lumberjacks": "Southland",
    # SWAC
    "Alabama A&M Bulldogs": "SWAC",
    "Alabama State Hornets": "SWAC",
    "Alcorn State Braves": "SWAC",
    "Bethune-Cookman Wildcats": "SWAC",
    "Florida A&M Rattlers": "SWAC",
    "Grambling Tigers": "SWAC",
    "Jackson State Tigers": "SWAC",
    "Mississippi Valley State Delta Devils": "SWAC",
    "Prairie View A&M Panthers": "SWAC",
    "Southern Jaguars": "SWAC",
    "Texas Southern Tigers": "SWAC",
    "Arkansas-Pine Bluff Golden Lions": "SWAC",
    # MEAC
    "Coppin State Eagles": "MEAC",
    "Delaware State Hornets": "MEAC",
    "Howard Bison": "MEAC",
    "Maryland Eastern Shore Hawks": "MEAC",
    "Morgan State Bears": "MEAC",
    "North Carolina A&T Aggies": "MEAC",
    "North Carolina Central Eagles": "MEAC",
    "Norfolk State Spartans": "MEAC",
    "South Carolina State Bulldogs": "MEAC",
    # NEC
    "Bryant Bulldogs": "NEC",
    "Central Connecticut Blue Devils": "NEC",
    "Fairleigh Dickinson Knights": "NEC",
    "Le Moyne Dolphins": "NEC",
    "Long Island University Sharks": "NEC",
    "Merrimack Warriors": "NEC",
    "Mount St. Mary's Mountaineers": "NEC",
    "Sacred Heart Pioneers": "NEC",
    "Saint Francis Red Flash": "NEC",
    "Stonehill Skyhawks": "NEC",
    "Wagner Seahawks": "NEC",
    # Summit League
    "Denver Pioneers": "Summit",
    "Kansas City Roos": "Summit",
    "North Dakota Fighting Hawks": "Summit",
    "North Dakota State Bison": "Summit",
    "Omaha Mavericks": "Summit",
    "South Dakota Coyotes": "Summit",
    "South Dakota State Jackrabbits": "Summit",
    "St. Thomas-Minnesota Tommies": "Summit",
    "Western Illinois Leathernecks": "Summit",
    # Big West
    "Cal Poly Mustangs": "Big West",
    "Cal State Bakersfield Roadrunners": "Big West",
    "Cal State Fullerton Titans": "Big West",
    "Cal State Northridge Matadors": "Big West",
    "California Baptist Lancers": "Big West",
    "Long Beach State Beach": "Big West",
    "UC Davis Aggies": "Big West",
    "UC Irvine Anteaters": "Big West",
    "UC Riverside Highlanders": "Big West",
    "UC San Diego Tritons": "Big West",
    "UC Santa Barbara Gauchos": "Big West",
    # WAC
    "Chicago State Cougars": "WAC",
    "Grand Canyon Lopes": "WAC",
    "Southern Utah Thunderbirds": "WAC",
    "Tarleton State Texans": "WAC",
    "Utah Tech Trailblazers": "WAC",
    "Utah Valley Wolverines": "WAC",
    "West Georgia Wolves": "WAC",
    # America East
    "UAlbany Great Danes": "America East",
    "Binghamton Bearcats": "America East",
    "Maine Black Bears": "America East",
    "New Hampshire Wildcats": "America East",
    "UMass Lowell River Hawks": "America East",
    "Vermont Catamounts": "America East",
    # ASUN
    "Bellarmine Knights": "ASUN",
    "Central Arkansas Bears": "ASUN",
    "Florida Gulf Coast Eagles": "ASUN",
    "Jacksonville Dolphins": "ASUN",
    "Kennesaw State Owls": "ASUN",
    "Lipscomb Bisons": "ASUN",
    "North Alabama Lions": "ASUN",
    "North Florida Ospreys": "ASUN",
    "Queens University Royals": "ASUN",
    "Stetson Hatters": "ASUN",
    "Lindenwood Lions": "ASUN",
    # SoCon / Southern extras
    "UNC Greensboro Spartans": "Southern",
    # MVC extras
    "SIU Edwardsville Cougars": "Missouri Valley",
    # Independent / misc
    "New Haven Chargers": "NEC",
    "Little Rock Trojans": "Sun Belt",
    "South Carolina Upstate Spartans": "Big South",
    "Southern Indiana Screaming Eagles": "OVC",
    "Incarnate Word Cardinals": "Southland",
}

# ── Avatar colors — deterministic from athlete_id ─────────────────────────────
AVATAR_COLORS = [
    "#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981",
    "#ef4444", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]

def avatar_color(athlete_id: str) -> str:
    return AVATAR_COLORS[int(athlete_id) % len(AVATAR_COLORS)]


# ── Helper functions ───────────────────────────────────────────────────────────
def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val and val.strip() else default
    except (ValueError, TypeError):
        return default

def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val)) if val and val.strip() else default
    except (ValueError, TypeError):
        return default

def inches_to_height_str(inches: int) -> str:
    feet = inches // 12
    remaining = inches % 12
    return f"{feet}'{remaining}\""

def infer_position(position_id: str, height_in: int) -> str:
    """
    ESPN position_id mapping for college basketball:
      1 = C  (avg 83", pure centers)
      2 = F  (avg 80", forwards — PF or C depending on height)
      3 = G/F (avg 75.5", guards and wings — split by height)
      4 = G  (avg 73", pure guards)
    """
    pid = safe_int(position_id, 3)
    if pid == 1:
        return "C"
    if pid == 2:
        return "C" if height_in >= 83 else "PF"
    if pid == 4:
        return "PG"
    # pid == 3 (or 6/7/8 edge cases): split by height
    # 6'5"+ → SF (wing range), 6'1"–6'4" → SG, under 6'1" → PG
    if height_in >= 77:
        return "SF"
    if height_in >= 73:
        return "SG"
    return "PG"

def experience_to_class_year(exp: str) -> str:
    e = safe_int(exp, 1)
    return {1: "FR", 2: "SO", 3: "JR", 4: "SR"}.get(e, "SR")

def eligibility_remaining(exp: str) -> int:
    e = safe_int(exp, 1)
    return max(1, 5 - e)

def compute_efg_pct(fgm: float, fg3m: float, fga: float) -> float:
    if fga <= 0:
        return 0.0
    return round((fgm + 0.5 * fg3m) / fga * 100, 1)

def compute_ts_pct(points: float, fga: float, fta: float) -> float:
    denom = 2 * (fga + 0.44 * fta)
    if denom <= 0:
        return 0.0
    return round(points / denom * 100, 1)

def compute_usage_pct(
    fga: float, fta: float, tov: float,
    team_fga: float, team_fta: float, team_tov: float, team_min: float, player_min: float,
) -> float:
    """Standard usage % = 100 * (FGA + 0.44*FTA + TOV) / (MP/TM_MP * (TM_FGA + 0.44*TM_FTA + TM_TOV))"""
    player_usage = fga + 0.44 * fta + tov
    team_usage = team_fga + 0.44 * team_fta + team_tov
    if team_usage <= 0 or team_min <= 0:
        return 0.0
    return round(100 * player_usage / (player_min / team_min * team_usage), 1)

def nil_estimate(ppg: float, position: str) -> list[int]:
    """Rough NIL range based on scoring tier and position."""
    is_big = position in ("PF", "C")
    if ppg >= 20:
        return [250_000, 600_000]
    if ppg >= 16:
        return [120_000, 300_000]
    if ppg >= 12:
        return [60_000, 150_000]
    if ppg >= 8:
        return [25_000, 80_000]
    if ppg >= 4:
        return [8_000, 30_000]
    # Bigs with low PPG can still command more for rebounding/defense
    return [5_000, 20_000] if is_big else [3_000, 12_000]


# ── CSV loading ────────────────────────────────────────────────────────────────
def load_transfer_ids(csv_path: Path = TRANSFERS_CSV) -> set[str]:
    """Return the set of athlete_ids found in a transfers CSV."""
    ids: set[str] = set()
    if not csv_path.exists():
        return ids
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("athlete_id", "").strip()
            if aid:
                ids.add(aid)
    return ids

def load_players(
    players_csv: Path = ALL_PLAYERS_CSV,
    transfers_csv: Path = TRANSFERS_CSV,
) -> list[Player]:
    if not players_csv.exists():
        return []

    transfer_ids = load_transfer_ids(transfers_csv)
    players: list[Player] = []

    with open(players_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            athlete_id = row.get("athlete_id", "").strip()
            if not athlete_id:
                continue

            # ── Core bio ──────────────────────────────────────────────────────
            name        = row.get("full_name", "").strip()
            school      = row.get("team_display_name", "").strip()
            headshot    = row.get("headshot_href", "").strip() or None
            height_in   = safe_int(row.get("height", "72"))
            weight_lbs  = safe_int(row.get("weight", "200"))
            position_id = row.get("position_id", "3").strip()
            experience  = row.get("experience_years", "1").strip()
            position    = infer_position(position_id, height_in)

            # ── Per-game stats ────────────────────────────────────────────────
            ppg  = safe_float(row.get("offensive_avg_points", "0"))
            rpg  = safe_float(row.get("general_avg_rebounds", "0"))
            apg  = safe_float(row.get("offensive_avg_assists", "0"))
            spg  = safe_float(row.get("defensive_avg_steals", "0"))
            bpg  = safe_float(row.get("defensive_avg_blocks", "0"))
            topg = safe_float(row.get("offensive_avg_turnovers", "0"))
            mpg  = safe_float(row.get("general_avg_minutes", "0"))
            # Extra per-game fields for per-40 fit score computation
            fga_pg   = safe_float(row.get("offensive_avg_field_goals_attempted", "0"))
            fta_pg   = safe_float(row.get("offensive_avg_free_throws_attempted", "0"))
            fg3m_pg  = safe_float(row.get("offensive_avg_three_point_field_goals_made", "0"))
            oreb_pg  = safe_float(row.get("offensive_avg_offensive_rebounds", "0"))
            dreb_pg  = safe_float(row.get("defensive_avg_defensive_rebounds", "0"))
            fouls_pg = safe_float(row.get("general_avg_fouls", "0"))

            # ── Shooting percentages (already 0–100 scale in CSV) ─────────────
            fg_pct  = safe_float(row.get("offensive_field_goal_pct", "0"))
            fg3_pct = safe_float(row.get("offensive_three_point_field_goal_pct", "0"))
            ft_pct  = safe_float(row.get("offensive_free_throw_pct", "0"))

            # ── Derived shooting metrics ──────────────────────────────────────
            fgm  = safe_float(row.get("offensive_field_goals_made", "0"))
            fga  = safe_float(row.get("offensive_field_goals_attempted", "0"))
            fg3m = safe_float(row.get("offensive_three_point_field_goals_made", "0"))
            fta  = safe_float(row.get("offensive_free_throws_attempted", "0"))
            pts  = safe_float(row.get("offensive_points", "0"))

            efg_pct = compute_efg_pct(fgm, fg3m, fga)
            ts_pct  = compute_ts_pct(pts, fga, fta)

            # ── Usage % ───────────────────────────────────────────────────────
            tov         = safe_float(row.get("offensive_turnovers", "0"))
            team_fga    = safe_float(row.get("team_stat_offensive_avg_field_goals_attempted", "0"))
            team_fta    = safe_float(row.get("team_stat_offensive_avg_free_throws_attempted", "0"))
            team_tov    = safe_float(row.get("team_stat_offensive_avg_turnovers", "0"))
            team_min    = safe_float(row.get("team_stat_general_avg_minutes", "200"))
            gp          = safe_float(row.get("general_games_played", "1"))
            usage = compute_usage_pct(
                fga / max(gp, 1), ft_pct / max(gp, 1), tov / max(gp, 1),
                team_fga, team_fta, team_tov, team_min, mpg,
            )
            # Simpler, more stable formula using per-game averages directly
            player_plays = (fga / max(gp, 1)) + 0.44 * (fta / max(gp, 1)) + topg
            team_plays   = team_fga + 0.44 * team_fta + team_tov
            usage = round(
                100 * player_plays / (mpg / max(team_min, 1) * team_plays), 1
            ) if team_plays > 0 and team_min > 0 else 0.0
            usage = max(0.0, min(50.0, usage))

            # ── Advanced metrics ──────────────────────────────────────────────
            per          = safe_float(row.get("general_per", "0"))
            bpm          = safe_float(row.get("box_plus_minus", "0"))
            bpm_scoring  = safe_float(row.get("bpm_scoring", "0"))
            bpm_defense  = safe_float(row.get("bpm_defense", "0"))

            # Approximate ortg/drtg from BPM components
            ortg = round(100 + bpm_scoring * 3.0, 1)
            drtg = round(100 - bpm_defense * 3.0, 1)

            # Rough win shares estimate
            win_shares = round(per / 15.0 * gp / 30.0, 2)

            # ── Classification & eligibility ──────────────────────────────────
            class_year     = experience_to_class_year(experience)
            elig_remaining = eligibility_remaining(experience)

            # ── Portal status ─────────────────────────────────────────────────
            is_portal = athlete_id in transfer_ids
            portal_status  = "available" if is_portal else "not_in_portal"
            portal_date    = "2025-04-03" if is_portal else ""

            # ── Conference ────────────────────────────────────────────────────
            conference = CONFERENCE_MAP.get(school, "Other")

            player = Player(
                id=athlete_id,
                name=name,
                position=position,
                previousSchool=school,
                conference=conference,
                classYear=class_year,
                eligibilityRemaining=elig_remaining,
                height=inches_to_height_str(height_in),
                weight=weight_lbs,
                hometown="",
                portalEntryDate=portal_date,
                portalStatus=portal_status,
                avatarColor=avatar_color(athlete_id),
                nilEstimate=nil_estimate(ppg, position),
                notes=[],
                isReturner=False,
                photoUrl=headshot,
                # Stats
                ppg=round(ppg, 1),
                rpg=round(rpg, 1),
                apg=round(apg, 1),
                spg=round(spg, 1),
                bpg=round(bpg, 1),
                topg=round(topg, 1),
                fgPct=round(fg_pct, 1),
                fg3Pct=round(fg3_pct, 1),
                ftPct=round(ft_pct, 1),
                efgPct=efg_pct,
                tsPct=ts_pct,
                usagePct=usage,
                ortg=ortg,
                drtg=drtg,
                bpm=round(bpm, 2),
                winShares=win_shares,
                per=round(per, 1),
                minutesPerGame=round(mpg, 1),
                fgaPerGame=round(fga_pg, 2),
                ftaPerGame=round(fta_pg, 2),
                fg3mPerGame=round(fg3m_pg, 2),
                orebPerGame=round(oreb_pg, 2),
                drebPerGame=round(dreb_pg, 2),
                foulsPerGame=round(fouls_pg, 2),
            )
            players.append(player)

            # Cache raw feature vector for ML predictor
            raw_feat = {f: safe_float(row.get(f, "0")) for f in PREDICTOR_FEATURES}
            raw_feat["weight"] = float(weight_lbs)
            raw_feat["height"] = float(height_in)
            _raw_features_cache[athlete_id] = raw_feat

    return players


# ── Load data at startup ───────────────────────────────────────────────────────
_mens_cache: list[Player] = []
_womens_cache: list[Player] = []


def _get_cache(gender: str) -> list[Player]:
    return _womens_cache if gender == "womens" else _mens_cache


@app.on_event("startup")
async def startup_event():
    global _mens_cache, _womens_cache
    _mens_cache = load_players()
    _womens_cache = load_players(WOMENS_ALL_PLAYERS_CSV, WOMENS_TRANSFERS_CSV)
    mens_transfer_ids = load_transfer_ids()
    womens_transfer_ids = load_transfer_ids(WOMENS_TRANSFERS_CSV)
    mens_portal = sum(1 for p in _mens_cache if p.id in mens_transfer_ids)
    womens_portal = sum(1 for p in _womens_cache if p.id in womens_transfer_ids)
    print(f"Loaded {len(_mens_cache)} men's players ({mens_portal} in transfer portal)")
    print(f"Loaded {len(_womens_cache)} women's players ({womens_portal} in transfer portal)")
    try:
        _load_predictor()
        print(f"Predictor loaded from {PREDICTOR_DIR}")
    except Exception as exc:
        print(f"Warning: predictor not available — {exc}")
    try:
        _load_team_predictor()
        print(f"Team record predictor loaded from {TEAM_PREDICTOR_DIR}")
    except Exception as exc:
        print(f"Warning: team record predictor not available — {exc}")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/players", response_model=list[Player])
def get_players(gender: str = Query("mens")):
    """Return all D1 players. Transfer portal players have portalEntryDate set."""
    return _get_cache(gender)

@app.get("/api/players/transfers", response_model=list[Player])
def get_transfers(gender: str = Query("mens")):
    """Return only transfer portal players (subset of /api/players)."""
    return [p for p in _get_cache(gender) if p.portalEntryDate != ""]

@app.get("/api/players/search", response_model=SearchResult)
def search_players(
    q:               Optional[str] = None,
    positions:       Optional[str] = None,   # comma-separated: PG,SG,SF,PF,C
    conferences:     Optional[str] = None,   # comma-separated conference names
    class_years:     Optional[str] = None,   # comma-separated: FR,SO,JR,SR,GRAD
    portal_statuses: Optional[str] = None,   # comma-separated
    ppg_min:  float = 0,
    ppg_max:  float = 100,
    fg3_min:  float = 0,
    fg3_max:  float = 100,
    efg_min:  float = 0,
    efg_max:  float = 100,
    min_elig: int   = 1,
    min_height: int = 60,
    max_height: int = 96,
    sort:  str = "ppg",         # ppg|rpg|apg|fg3Pct|efgPct|name|portalEntryDate
    offset: int = Query(0, ge=0),
    limit:  int = Query(48, ge=1, le=200),
    gender: str = Query("mens"),
):
    """
    Server-side filtered + paginated player search.
    Used by the Explore page — returns a small batch instead of all 3k+ players.
    """
    results = _get_cache(gender)

    # ── Text search ───────────────────────────────────────────────────────────
    if q:
        ql = q.lower()
        results = [
            p for p in results
            if ql in p.name.lower()
            or ql in p.previousSchool.lower()
            or ql in p.conference.lower()
            or ql in p.position.lower()
        ]

    # ── Category filters ──────────────────────────────────────────────────────
    if positions:
        pos_set = {p.strip() for p in positions.split(",")}
        results = [p for p in results if p.position in pos_set]

    if conferences:
        conf_set = {c.strip() for c in conferences.split(",")}
        results = [p for p in results if p.conference in conf_set]

    if class_years:
        cy_set = {c.strip() for c in class_years.split(",")}
        results = [p for p in results if p.classYear in cy_set]

    if portal_statuses:
        ps_set = {s.strip() for s in portal_statuses.split(",")}
        results = [p for p in results if p.portalStatus in ps_set]

    # ── Stat range filters ────────────────────────────────────────────────────
    def height_in(h: str) -> int:
        try:
            feet, rest = h.split("'")
            inches = rest.replace('"', '').strip()
            return int(feet) * 12 + int(inches)
        except Exception:
            return 72

    results = [
        p for p in results
        if ppg_min <= p.ppg <= ppg_max
        and fg3_min <= p.fg3Pct <= fg3_max
        and efg_min <= p.efgPct <= efg_max
        and p.eligibilityRemaining >= min_elig
        and min_height <= height_in(p.height) <= max_height
    ]

    # ── Sort ──────────────────────────────────────────────────────────────────
    sort_map = {
        "ppg":             lambda p: -p.ppg,
        "rpg":             lambda p: -p.rpg,
        "apg":             lambda p: -p.apg,
        "fg3Pct":          lambda p: -p.fg3Pct,
        "efgPct":          lambda p: -p.efgPct,
        "name":            lambda p: p.name.lower(),
        "portalEntryDate": lambda p: p.portalEntryDate or "",
    }
    key_fn = sort_map.get(sort, sort_map["ppg"])
    results = sorted(results, key=key_fn)

    # ── Paginate ──────────────────────────────────────────────────────────────
    total = len(results)
    page_players = results[offset: offset + limit]

    return SearchResult(total=total, offset=offset, limit=limit, players=page_players)


def _run_prediction(player_id: str, projected_mpg: float, raw: dict) -> PredictionResult:
    """Build the ordered feature list, call the predictor, and return a PredictionResult."""
    feature_list = [raw.get(f, 0.0) for f in PREDICTOR_FEATURES] + [projected_mpg]
    raw_results = _predictor.run_prediction(feature_list)
    pred = {name: max(0.0, float(value)) for name, value in raw_results}

    return PredictionResult(
        player_id=player_id,
        projected_mpg=projected_mpg,
        # Scoring
        points=round(pred.get("next_offensive_avg_points", 0.0), 2),
        fg_made=round(pred.get("next_offensive_avg_field_goals_made", 0.0), 2),
        fg_attempted=round(pred.get("next_offensive_avg_field_goals_attempted", 0.0), 2),
        fg_pct=round(pred.get("next_offensive_field_goal_pct", 0.0), 4),
        fg3_made=round(pred.get("next_offensive_avg_three_point_field_goals_made", 0.0), 2),
        fg3_attempted=round(pred.get("next_offensive_avg_three_point_field_goals_attempted", 0.0), 2),
        fg3_pct=round(pred.get("next_offensive_three_point_field_goal_pct", 0.0), 4),
        fg2_made=round(pred.get("next_offensive_avg_two_point_field_goals_made", 0.0), 2),
        fg2_attempted=round(pred.get("next_offensive_avg_two_point_field_goals_attempted", 0.0), 2),
        fg2_pct=round(pred.get("next_offensive_two_point_field_goal_pct", 0.0), 4),
        ft_made=round(pred.get("next_offensive_avg_free_throws_made", 0.0), 2),
        ft_attempted=round(pred.get("next_offensive_avg_free_throws_attempted", 0.0), 2),
        ft_pct=round(pred.get("next_offensive_free_throws", 0.0), 4),
        # Playmaking
        assists=round(pred.get("next_offensive_avg_assists", 0.0), 2),
        turnovers=round(pred.get("next_offensive_avg_turnovers", 0.0), 2),
        # Rebounding
        offensive_rebounds=round(pred.get("next_offensive_avg_offensive_rebounds", 0.0), 2),
        defensive_rebounds=round(pred.get("next_defensive_avg_defensive_rebounds", 0.0), 2),
        total_rebounds=round(pred.get("next_general_avg_rebounds", 0.0), 2),
        # Defense
        steals=round(pred.get("next_defensive_avg_steals", 0.0), 2),
        blocks=round(pred.get("next_defensive_avg_blocks", 0.0), 2),
        fouls=round(pred.get("next_general_avg_fouls", 0.0), 2),
        # Efficiency
        shooting_efficiency=round(pred.get("next_offensive_shooting_efficiency", 0.0), 4),
        scoring_efficiency=round(pred.get("next_offensive_scoring_efficiency", 0.0), 4),
    )


@app.get("/api/players/{player_id}/prediction", response_model=PredictionResult)
def get_player_prediction(
    player_id: str,
    projected_mpg: float = Query(..., description="Projected minutes per game for next season"),
):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not available")
    raw = _raw_features_cache.get(player_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Player not found")
    return _run_prediction(player_id, projected_mpg, raw)


@app.post("/api/players/predictions/batch", response_model=list[PredictionResult])
def batch_predictions(request: BatchPredictionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not available")
    results: list[PredictionResult] = []
    for item in request.players:
        if item.projected_mpg is None:
            continue
        raw = _raw_features_cache.get(item.player_id)
        if not raw:
            continue
        results.append(_run_prediction(item.player_id, item.projected_mpg, raw))
    return results


@app.get("/api/players/{player_id}", response_model=Player)
def get_player(player_id: str, gender: str = Query("mens")):
    from fastapi import HTTPException
    player = next((p for p in _get_cache(gender) if p.id == player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@app.get("/health")
def health():
    return {"status": "ok", "mens_loaded": len(_mens_cache), "womens_loaded": len(_womens_cache)}


# ── Team list endpoint ─────────────────────────────────────────────────────────

@app.get("/api/teams")
def get_teams():
    """Return all NCAA D1 teams with their conference, sorted alphabetically."""
    import pandas as pd
    csv_path = TEAM_PREDICTOR_DIR / "2026 teams data set.csv"
    df = pd.read_csv(csv_path)
    teams = [{"team": row["TEAM"], "conference": row["CONF"]} for _, row in df.iterrows()]
    return sorted(teams, key=lambda x: x["team"])


# ── Conference record prediction ───────────────────────────────────────────────

class ConferencePredictRequest(BaseModel):
    team_name: str
    player_bpm: list[float]
    player_minutes: list[float] | None = None
    simulations: int = 10_000


@app.post("/api/conference-predict")
def predict_conference(req: ConferencePredictRequest):
    """
    Run the BPM-based conference win probability + Monte Carlo season simulation.
    Returns win_probabilities (per opponent) and monte_carlo distribution.
    """
    if _team_predictor is None:
        raise HTTPException(status_code=503, detail="Team record predictor not available")
    try:
        result = _team_predictor.conference_win_probabilities(
            team_name=req.team_name,
            team_bpm=req.player_bpm,
            team_minutes=req.player_minutes,
            simulations=req.simulations,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Team context endpoint ──────────────────────────────────────────────────────

class TeamSeasonStats(BaseModel):
    team: str
    conference: str
    games: int
    wins: int
    adj_oe: float
    adj_de: float
    net_rating: float
    barthag: float
    efg_o: float
    efg_d: float
    tor: float
    tord: float
    orb: float
    drb: float
    ftr: float
    ftrd: float
    two_p_o: float
    two_p_d: float
    three_p_o: float
    three_p_d: float
    adj_t: float
    wab: float
    conference_rank: int   # rank within conference for net_rating (1 = best)
    conference_size: int   # total teams in conference


@app.get("/api/team-context/{team_name}", response_model=dict)
def get_team_context(team_name: str):
    """
    Return last-year KenPom-style stats for the requested team plus the
    average of all teams in the same conference. Used by the Team Analytics tabs.
    """
    csv_path = TEAM_PREDICTOR_DIR / "2026 teams data set.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=503, detail="Teams dataset not available")

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Case-insensitive match
    team_row = next(
        (r for r in rows if r["TEAM"].strip().lower() == team_name.strip().lower()),
        None,
    )
    if team_row is None:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found in dataset")

    conference = team_row["CONF"].strip()
    conf_rows = [r for r in rows if r["CONF"].strip() == conference]

    def row_to_stats(r: dict, conf: str, rank: int = 0, conf_size: int = 0) -> dict:
        adj_oe = safe_float(r.get("ADJOE", "0"))
        adj_de = safe_float(r.get("ADJDE", "0"))
        return {
            "team": r["TEAM"].strip(),
            "conference": conf,
            "games": safe_int(r.get("G", "0")),
            "wins": safe_int(r.get("W", "0")),
            "adj_oe": round(adj_oe, 1),
            "adj_de": round(adj_de, 1),
            "net_rating": round(adj_oe - adj_de, 1),
            "barthag": round(safe_float(r.get("BARTHAG", "0")), 4),
            "efg_o": round(safe_float(r.get("EFG_O", "0")), 1),
            "efg_d": round(safe_float(r.get("EFG_D", "0")), 1),
            "tor": round(safe_float(r.get("TOR", "0")), 1),
            "tord": round(safe_float(r.get("TORD", "0")), 1),
            "orb": round(safe_float(r.get("ORB", "0")), 1),
            "drb": round(safe_float(r.get("DRB", "0")), 1),
            "ftr": round(safe_float(r.get("FTR", "0")), 1),
            "ftrd": round(safe_float(r.get("FTRD", "0")), 1),
            "two_p_o": round(safe_float(r.get("2P_O", "0")), 1),
            "two_p_d": round(safe_float(r.get("2P_D", "0")), 1),
            "three_p_o": round(safe_float(r.get("3P_O", "0")), 1),
            "three_p_d": round(safe_float(r.get("3P_D", "0")), 1),
            "adj_t": round(safe_float(r.get("ADJ_T", "0")), 1),
            "wab": round(safe_float(r.get("WAB", "0")), 1),
            "conference_rank": rank,
            "conference_size": conf_size,
        }

    # Compute conference average
    def avg_field(field: str) -> float:
        vals = [safe_float(r.get(field, "0")) for r in conf_rows]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    avg_adj_oe = avg_field("ADJOE")
    avg_adj_de = avg_field("ADJDE")

    # Rank team within conference by net rating (descending)
    net_ratings = sorted(
        [(r["TEAM"].strip(), safe_float(r.get("ADJOE", "0")) - safe_float(r.get("ADJDE", "0"))) for r in conf_rows],
        key=lambda x: x[1], reverse=True,
    )
    team_rank = next((i + 1 for i, (t, _) in enumerate(net_ratings) if t.lower() == team_name.strip().lower()), 0)

    # Build a synthetic "average" row for the conference
    avg_row = {
        "TEAM": f"{conference} Average",
        "CONF": conference,
        "G": str(round(avg_field("G"))),
        "W": str(round(avg_field("W"))),
        "ADJOE": str(avg_adj_oe),
        "ADJDE": str(avg_adj_de),
        "BARTHAG": str(avg_field("BARTHAG")),
        "EFG_O": str(avg_field("EFG_O")),
        "EFG_D": str(avg_field("EFG_D")),
        "TOR": str(avg_field("TOR")),
        "TORD": str(avg_field("TORD")),
        "ORB": str(avg_field("ORB")),
        "DRB": str(avg_field("DRB")),
        "FTR": str(avg_field("FTR")),
        "FTRD": str(avg_field("FTRD")),
        "2P_O": str(avg_field("2P_O")),
        "2P_D": str(avg_field("2P_D")),
        "3P_O": str(avg_field("3P_O")),
        "3P_D": str(avg_field("3P_D")),
        "ADJ_T": str(avg_field("ADJ_T")),
        "WAB": str(avg_field("WAB")),
    }

    return {
        "team": row_to_stats(team_row, conference, rank=team_rank, conf_size=len(conf_rows)),
        "conference_avg": row_to_stats(avg_row, conference, rank=0, conf_size=len(conf_rows)),
        "conference_standings": [
            {"team": t, "net_rating": round(nr, 1)} for t, nr in net_ratings
        ],
    }


# ── Auth endpoints ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    data_view: str = "mens"
    team_name: str | None = None


@app.post("/api/auth/signup")
def signup(body: SignupRequest):
    """
    Create a new account with email already confirmed (bypasses Supabase
    email-verification flow). The frontend then signs in immediately with
    signInWithPassword so a live session is established.
    """
    try:
        user_id = create_confirmed_user(
            email=body.email,
            password=body.password,
            metadata={
                "full_name": body.full_name,
                "data_view": body.data_view,
                "team_name": body.team_name,
            },
        )
        return {"user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── User data endpoints ────────────────────────────────────────────────────────

@app.get("/api/user/data")
def get_user_data(user_id: str = Depends(get_current_user_id)):
    """Return all persisted data for the authenticated user."""
    return fetch_user_data(user_id)


@app.put("/api/user/scenarios/{scenario_id}")
def put_scenario(
    scenario_id: str,
    scenario: Any = Body(...),
    user_id: str = Depends(get_current_user_id),
):
    """Create or replace a scenario."""
    scenario["id"] = scenario_id
    upsert_scenario(user_id, scenario)
    return {"ok": True}


@app.delete("/api/user/scenarios/{scenario_id}")
def delete_scenario(
    scenario_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a scenario (ownership enforced)."""
    remove_scenario(user_id, scenario_id)
    return {"ok": True}


@app.put("/api/user/notes/{player_id}")
def put_notes(
    player_id: str,
    body: Any = Body(...),
    user_id: str = Depends(get_current_user_id),
):
    """Upsert the notes array for a single player."""
    notes = body.get("notes", []) if isinstance(body, dict) else []
    upsert_player_notes(user_id, player_id, notes)
    return {"ok": True}


@app.put("/api/user/models/{model_id}")
def put_model(
    model_id: str,
    model: Any = Body(...),
    user_id: str = Depends(get_current_user_id),
):
    """Create or replace a custom evaluation model."""
    model["id"] = model_id
    upsert_model(user_id, model)
    return {"ok": True}


@app.delete("/api/user/models/{model_id}")
def delete_model(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a custom evaluation model (ownership enforced)."""
    remove_model(user_id, model_id)
    return {"ok": True}


# ── Admin analytics ────────────────────────────────────────────────────────────

ANALYTICS_PASSWORD = "courtexai"

@app.get("/api/admin/analytics")
def get_analytics(x_analytics_key: str = Header(None)):
    """
    Admin-only analytics endpoint. Protected by a static key header.
    Returns aggregated user, roster, and NIL data from Supabase.
    """
    if x_analytics_key != ANALYTICS_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")

    sb = get_supabase()

    # ── Users ──────────────────────────────────────────────────────────────────
    try:
        users_response = sb.auth.admin.list_users()
        raw_users = users_response if isinstance(users_response, list) else []
    except Exception:
        raw_users = []

    user_details = []
    users_by_school: dict[str, int] = {}
    users_by_conference: dict[str, int] = {}

    for u in raw_users:
        meta = getattr(u, "user_metadata", {}) or {}
        school = meta.get("school") or "Unknown"
        conference = meta.get("conference") or "Unknown"
        users_by_school[school] = users_by_school.get(school, 0) + 1
        users_by_conference[conference] = users_by_conference.get(conference, 0) + 1
        user_details.append({
            "id":          str(getattr(u, "id", "")),
            "email":       getattr(u, "email", ""),
            "name":        meta.get("full_name", ""),
            "school":      school,
            "conference":  conference,
            "team_name":   meta.get("team_name", ""),
            "created_at":  str(getattr(u, "created_at", "")),
        })

    # ── Scenarios & NIL ────────────────────────────────────────────────────────
    scenarios_resp = sb.table("scenarios").select("*").execute()
    scenarios = scenarios_resp.data or []

    user_map = {u["id"]: u for u in user_details}

    nil_by_player: dict[str, list] = {}
    account_budgets = []

    for s in scenarios:
        uid = str(s.get("user_id", ""))
        user_info = user_map.get(uid, {})
        school = user_info.get("school", "Unknown")
        budget = s.get("budget") or 0
        nil_deals: dict = s.get("nil_deals") or {}

        committed = 0
        targeted = 0

        for player_id, deal in nil_deals.items():
            amount = deal.get("offerAmount") or 0
            deal_status = deal.get("status", "not_targeted")

            if deal_status == "signed":
                committed += amount
            elif deal_status in ("targeted", "offered", "negotiating"):
                targeted += amount

            if player_id not in nil_by_player:
                nil_by_player[player_id] = []
            nil_by_player[player_id].append({
                "school":   school,
                "scenario": s.get("name", ""),
                "amount":   amount,
                "status":   deal_status,
            })

        account_budgets.append({
            "user_id":       uid,
            "school":        school,
            "scenario_name": s.get("name", ""),
            "budget":        budget,
            "committed":     committed,
            "targeted":      targeted,
            "remaining":     budget - committed - targeted,
        })

    return {
        "user_count":         len(user_details),
        "roster_count":       len(scenarios),
        "users_by_school":    dict(sorted(users_by_school.items(), key=lambda x: -x[1])),
        "users_by_conference": dict(sorted(users_by_conference.items(), key=lambda x: -x[1])),
        "user_details":       sorted(user_details, key=lambda u: u["created_at"], reverse=True),
        "nil_by_player":      nil_by_player,
        "account_budgets":    account_budgets,
    }
