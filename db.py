"""
db.py — Supabase helper functions for Nexus Analytics
All Supabase interactions live here; main.py stays thin route handlers.
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

_bearer = HTTPBearer()

# ── Supabase admin client (service role — server-side only) ───────────────────

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Singleton Supabase client with service role key."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ── Auth dependency ───────────────────────────────────────────────────────────

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    FastAPI dependency — verifies the Supabase JWT sent by the frontend and
    returns the authenticated user's UUID.

    Raises HTTP 401 if the token is invalid or expired.
    """
    token = credentials.credentials
    sb = get_supabase()
    try:
        resp = sb.auth.get_user(token)
        if not resp.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return str(resp.user.id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


# ── User creation (admin — bypasses email confirmation) ───────────────────────

def create_confirmed_user(email: str, password: str, metadata: dict) -> str:
    """
    Create a new user via the admin API with email already confirmed.
    Returns the new user's UUID as a string.
    """
    sb = get_supabase()
    response = sb.auth.admin.create_user({
        "email": email,
        "password": password,
        "email_confirm": True,
        "user_metadata": metadata,
    })
    return str(response.user.id)


# ── User data bootstrap ───────────────────────────────────────────────────────

def fetch_user_data(user_id: str) -> dict:
    """
    Returns all persisted user data in one query set:
      scenarios, player_notes, models
    """
    sb = get_supabase()

    scenarios_resp = (
        sb.table("scenarios")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )

    notes_resp = (
        sb.table("player_notes")
        .select("player_id, notes")
        .eq("user_id", user_id)
        .execute()
    )

    models_resp = (
        sb.table("evaluation_models")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    scenarios = _rows_to_scenarios(scenarios_resp.data or [])
    player_notes = {row["player_id"]: row["notes"] for row in (notes_resp.data or [])}
    models = _rows_to_models(models_resp.data or [])

    return {
        "scenarios":    scenarios,
        "player_notes": player_notes,
        "models":       models,
    }


# ── Scenarios ─────────────────────────────────────────────────────────────────

def upsert_scenario(user_id: str, scenario: dict) -> None:
    """Create or fully replace a scenario row."""
    sb = get_supabase()
    sb.table("scenarios").upsert(
        _scenario_to_row(user_id, scenario),
        on_conflict="id",
    ).execute()


def remove_scenario(user_id: str, scenario_id: str) -> None:
    """Delete a scenario, enforcing ownership."""
    sb = get_supabase()
    sb.table("scenarios").delete().eq("id", scenario_id).eq("user_id", user_id).execute()


# ── Player notes ──────────────────────────────────────────────────────────────

def upsert_player_notes(user_id: str, player_id: str, notes: list[str]) -> None:
    """Upsert the notes array for a single player."""
    sb = get_supabase()
    sb.table("player_notes").upsert(
        {"user_id": user_id, "player_id": player_id, "notes": notes},
        on_conflict="user_id,player_id",
    ).execute()


# ── Evaluation models ─────────────────────────────────────────────────────────

def upsert_model(user_id: str, model: dict) -> None:
    """Create or replace a custom evaluation model."""
    sb = get_supabase()
    sb.table("evaluation_models").upsert(
        _model_to_row(user_id, model),
        on_conflict="id",
    ).execute()


def remove_model(user_id: str, model_id: str) -> None:
    """Delete a model, enforcing ownership."""
    sb = get_supabase()
    sb.table("evaluation_models").delete().eq("id", model_id).eq("user_id", user_id).execute()


# ── Row ↔ dict converters ─────────────────────────────────────────────────────

def _scenario_to_row(user_id: str, s: dict) -> dict:
    return {
        "id":            s["id"],
        "user_id":       user_id,
        "name":          s.get("name", "My Roster"),
        "budget":        s.get("budget", 5_000_000),
        "created_at":    s.get("createdAt"),
        "slots":         s.get("slots", []),
        "board_groups":  s.get("boardGroups", []),
        "roster_groups": s.get("rosterGroups", []),
        "watchlist_ids": s.get("watchlistIds", []),
        "nil_deals":     s.get("nilDeals", {}),
        "player_minutes": s.get("playerMinutes", {}),
    }


def _row_to_scenario(row: dict) -> dict:
    return {
        "id":            row["id"],
        "name":          row["name"],
        "budget":        row["budget"],
        "createdAt":     row.get("created_at", ""),
        "slots":         row.get("slots") or [],
        "boardGroups":   row.get("board_groups") or [],
        "rosterGroups":  row.get("roster_groups") or [],
        "watchlistIds":  row.get("watchlist_ids") or [],
        "nilDeals":      row.get("nil_deals") or {},
        "playerMinutes": row.get("player_minutes") or {},
    }


def _rows_to_scenarios(rows: list[dict]) -> list[dict]:
    return [_row_to_scenario(r) for r in rows]


def _model_to_row(user_id: str, m: dict) -> dict:
    return {
        "id":           m["id"],
        "user_id":      user_id,
        "name":         m.get("name", ""),
        "description":  m.get("description", ""),
        "coefficients": m.get("coefficients", {}),
        "is_preset":    m.get("isPreset", False),
        "created_at":   m.get("createdAt"),
    }


def _row_to_model(row: dict) -> dict:
    return {
        "id":           row["id"],
        "name":         row["name"],
        "description":  row.get("description", ""),
        "coefficients": row.get("coefficients") or {},
        "isPreset":     row.get("is_preset", False),
        "createdAt":    row.get("created_at", ""),
    }


def _rows_to_models(rows: list[dict]) -> list[dict]:
    return [_row_to_model(r) for r in rows]
