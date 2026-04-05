-- ─────────────────────────────────────────────────────────────────────────────
-- Nexus Analytics — Supabase schema
-- Run this in the Supabase SQL editor (Dashboard → SQL Editor → New query).
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable UUID extension (already enabled on Supabase by default)
create extension if not exists "uuid-ossp";

-- ── profiles ──────────────────────────────────────────────────────────────────
-- Mirrors auth.users; populated automatically via trigger on first sign-up.
create table if not exists public.profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  full_name   text,
  data_view   text default 'mens',   -- 'mens' | 'womens'
  team_name   text,                  -- NCAA D1 team the coach belongs to
  created_at  timestamptz default now()
);

alter table public.profiles enable row level security;

create policy "Users can view own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);

-- Auto-create a profile row when a new user signs up
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, full_name, data_view, team_name)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    coalesce(new.raw_user_meta_data->>'data_view', 'mens'),
    new.raw_user_meta_data->>'team_name'
  );
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- ── scenarios ─────────────────────────────────────────────────────────────────
create table if not exists public.scenarios (
  id              text        primary key,
  user_id         uuid        not null references auth.users(id) on delete cascade,
  name            text        not null default 'My Roster',
  budget          bigint      not null default 5000000,
  created_at      timestamptz default now(),
  slots           jsonb       not null default '[]',
  board_groups    jsonb       not null default '[]',
  roster_groups   jsonb       not null default '[]',
  watchlist_ids   jsonb       not null default '[]',
  nil_deals       jsonb       not null default '{}',
  player_minutes  jsonb       not null default '{}'
);

alter table public.scenarios enable row level security;

create policy "Users can manage own scenarios"
  on public.scenarios for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- ── player_notes ──────────────────────────────────────────────────────────────
create table if not exists public.player_notes (
  user_id    uuid  not null references auth.users(id) on delete cascade,
  player_id  text  not null,
  notes      jsonb not null default '[]',
  primary key (user_id, player_id)
);

alter table public.player_notes enable row level security;

create policy "Users can manage own notes"
  on public.player_notes for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- ── evaluation_models ─────────────────────────────────────────────────────────
create table if not exists public.evaluation_models (
  id            text        primary key,
  user_id       uuid        not null references auth.users(id) on delete cascade,
  name          text        not null default '',
  description   text        not null default '',
  coefficients  jsonb       not null default '{}',
  is_preset     boolean     not null default false,
  created_at    timestamptz default now()
);

alter table public.evaluation_models enable row level security;

create policy "Users can manage own models"
  on public.evaluation_models for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
