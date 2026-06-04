#!/usr/bin/env python3
"""WC2026 live predictions — Monte Carlo simulator.

Fetches live results from football-data.org, updates team strength
estimates via a Bayesian-style approach, then runs N Monte Carlo
simulations of the remaining tournament.

Usage:
    python src/live_updater.py           # live mode (calls API)
    python src/live_updater.py --demo    # demo mode (no API calls)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

FD_KEY  = os.getenv("FD_API_KEY", "")
N_SIM   = int(os.getenv("N_SIMULATIONS", "10000"))
OUT     = Path("output/live_predictions.json")
FD_BASE = "https://api.football-data.org/v4"

# Average goals per team per game (WC historical)
AVG_GOALS = 1.35

# ── Base team strengths (Elo-like, calibrated to FIFA rankings ~Apr-2026) ─────
BASE_STRENGTH: dict[str, float] = {
    "France": 1.82, "Brazil": 1.78, "England": 1.75, "Germany": 1.73,
    "Spain": 1.71, "Argentina": 1.68, "Portugal": 1.63, "Netherlands": 1.58,
    "Belgium": 1.55, "Italy": 1.52, "Colombia": 1.48, "Uruguay": 1.45,
    "Mexico": 1.42, "United States": 1.40, "Croatia": 1.38, "Morocco": 1.35,
    "Japan": 1.32, "Switzerland": 1.30, "South Korea": 1.28, "Norway": 1.27,
    "Sweden": 1.25, "Denmark": 1.24, "Ecuador": 1.22, "Senegal": 1.20,
    "Australia": 1.18, "Poland": 1.17, "Canada": 1.16, "Peru": 1.15,
    "Chile": 1.14, "Algeria": 1.13, "Egypt": 1.12, "Tunisia": 1.11,
    "Costa Rica": 1.10, "Panama": 1.09, "Venezuela": 1.08, "Ivory Coast": 1.07,
    "Nigeria": 1.06, "Ghana": 1.05, "Jamaica": 1.04, "Honduras": 1.03,
    "Saudi Arabia": 1.02, "Iran": 1.01, "South Africa": 1.00, "Iceland": 0.99,
    "New Zealand": 0.98, "Indonesia": 0.97, "Guatemala": 0.96, "Bolivia": 0.95,
}

# football-data.org team name → our canonical name
NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "United States": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "New Zealand": "New Zealand",
    "Saudi Arabia": "Saudi Arabia",
}


def normalize(name: str) -> str:
    return NAME_MAP.get(name, name)


# ── API helpers ───────────────────────────────────────────────────────────────

def fd_get(path: str) -> dict | None:
    if not FD_KEY:
        print("⚠️  FD_API_KEY not set", file=sys.stderr)
        return None
    try:
        r = requests.get(
            f"{FD_BASE}{path}",
            headers={"X-Auth-Token": FD_KEY},
            timeout=20,
        )
        if r.status_code == 429:
            print("⚠️  Rate limit hit on football-data.org", file=sys.stderr)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️  API error ({path}): {e}", file=sys.stderr)
        return None


def fetch_wc_matches() -> list[dict]:
    data = fd_get("/competitions/WC/matches?limit=200")
    if not data:
        return []
    matches = []
    for m in data.get("matches", []):
        home = normalize(m.get("homeTeam", {}).get("name", ""))
        away = normalize(m.get("awayTeam", {}).get("name", ""))
        if not home or not away:
            continue
        score = m.get("score", {}).get("fullTime", {})
        matches.append({
            "home": home,
            "away": away,
            "stage": m.get("stage", ""),
            "group": m.get("group", "") or "",
            "status": m.get("status", "SCHEDULED"),
            "home_score": score.get("home"),
            "away_score": score.get("away"),
            "date": m.get("utcDate", ""),
        })
    return matches


# ── Date formatting ───────────────────────────────────────────────────────────

_DAYS   = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
_MONTHS = ["", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
           "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _fmt_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return f"{_DAYS[dt.weekday()]} {dt.day} {_MONTHS[dt.month]}"
    except Exception:
        return iso[:10]


def _fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return f"{dt.hour:02d}:{dt.minute:02d}"
    except Exception:
        return "—"


def next_match_info(matches: list[dict]) -> dict:
    live = [m for m in matches if m["status"] in ("LIVE", "IN_PLAY", "PAUSED")]
    if live:
        m = live[0]
        return {
            "team1": m["home"], "team2": m["away"],
            "actual_score": f"{m['home_score'] or 0}-{m['away_score'] or 0}",
            "pred_score":   f"{m['home_score'] or 0}-{m['away_score'] or 0}",
            "date": _fmt_date(m["date"]), "time": _fmt_time(m["date"]),
            "group": m["group"].replace("GROUP_", "") if m["group"] else "KO",
            "p1": 50, "status": "live",
        }
    scheduled = [m for m in matches if m["status"] == "SCHEDULED"]
    if scheduled:
        m = scheduled[0]
        return {
            "team1": m["home"], "team2": m["away"],
            "pred_score": "—",
            "date": _fmt_date(m["date"]), "time": _fmt_time(m["date"]),
            "group": m["group"].replace("GROUP_", "") if m["group"] else "KO",
            "p1": 50, "status": "scheduled",
        }
    # Default pre-tournament
    return {
        "team1": "Mexico", "team2": "South Africa",
        "pred_score": "2-0", "date": "Jue 11 Jun", "time": "16:00",
        "group": "A", "p1": 64, "status": "scheduled",
    }


# ── Strength update ───────────────────────────────────────────────────────────

def update_strengths(
    matches: list[dict],
    base: dict[str, float],
) -> dict[str, float]:
    """Elo-style update: adjusts strengths proportional to result surprise."""
    strengths = dict(base)
    k = 0.04

    for m in matches:
        if m["status"] != "FINISHED":
            continue
        h, a = m["home"], m["away"]
        hs, as_ = m["home_score"], m["away_score"]
        if hs is None or as_ is None:
            continue

        sh = strengths.get(h, 1.1)
        sa = strengths.get(a, 1.1)
        expected_h = sh / (sh + sa)
        actual_h   = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        surprise   = actual_h - expected_h

        strengths[h] = max(0.5, sh + k * surprise)
        strengths[a] = max(0.5, sa - k * surprise)

    return strengths


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def _sim_match(
    s1: float, s2: float, rng: np.random.Generator
) -> tuple[int, int]:
    lam1 = AVG_GOALS * s1 / (s1 + s2) * 2
    lam2 = AVG_GOALS * s2 / (s1 + s2) * 2
    return int(rng.poisson(lam1)), int(rng.poisson(lam2))


def _sim_knockout(
    t1: str, t2: str, s: dict[str, float], rng: np.random.Generator
) -> str:
    g1, g2 = _sim_match(s.get(t1, 1.1), s.get(t2, 1.1), rng)
    if g1 != g2:
        return t1 if g1 > g2 else t2
    # Penalty shootout weighted by relative strength
    p1 = s.get(t1, 1.1) / (s.get(t1, 1.1) + s.get(t2, 1.1))
    return t1 if rng.random() < p1 else t2


def _sim_group(
    teams: list[str],
    s: dict[str, float],
    rng: np.random.Generator,
    fixed: dict[tuple, tuple],
) -> list[str]:
    """Returns teams sorted by (pts desc, gd desc, gf desc)."""
    pts: dict[str, int] = defaultdict(int)
    gd:  dict[str, int] = defaultdict(int)
    gf:  dict[str, int] = defaultdict(int)

    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            if (t1, t2) in fixed:
                g1, g2 = fixed[(t1, t2)]
            elif (t2, t1) in fixed:
                g2, g1 = fixed[(t2, t1)]
            else:
                g1, g2 = _sim_match(s.get(t1, 1.1), s.get(t2, 1.1), rng)

            gf[t1] += g1; gf[t2] += g2
            gd[t1] += g1 - g2; gd[t2] += g2 - g1

            if g1 > g2:
                pts[t1] += 3
            elif g2 > g1:
                pts[t2] += 3
            else:
                pts[t1] += 1; pts[t2] += 1

    return sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))


def _simulate_one(
    groups: dict[str, list[str]],
    strengths: dict[str, float],
    fixed_group: dict[str, dict[tuple, tuple]],
    rng: np.random.Generator,
) -> dict[str, int]:
    """
    Simulate one full tournament.
    Returns {team: round_eliminated} where 0 = champion,
    1 = finalist, 2 = semifinalist, etc.
    """
    finish: dict[str, int] = {}

    # ── Group stage ──────────────────────────────────────────────────────────
    thirds: list[tuple[float, str]] = []  # (strength, team)

    bracket32: list[str] = []
    for gname, teams in sorted(groups.items()):
        ranked = _sim_group(teams, strengths, rng, fixed_group.get(gname, {}))

        bracket32.append(ranked[0])  # 1st
        bracket32.append(ranked[1])  # 2nd

        if len(ranked) >= 3:
            thirds.append((strengths.get(ranked[2], 1.0), ranked[2]))
        if len(ranked) >= 4:
            for t in ranked[3:]:
                finish[t] = 7  # group stage exit

    # 8 best 3rd-place teams advance
    thirds.sort(key=lambda x: -x[0])
    for _, t in thirds[:8]:
        bracket32.append(t)
    for _, t in thirds[8:]:
        finish[t] = 7

    # Ensure exactly 32
    bracket32 = bracket32[:32]

    # ── Knockout rounds ───────────────────────────────────────────────────────
    current = list(bracket32)
    for round_num in range(5, 0, -1):  # 5=R32 … 1=Final
        next_round: list[str] = []
        for i in range(0, len(current), 2):
            if i + 1 >= len(current):
                next_round.append(current[i])
                continue
            t1, t2 = current[i], current[i + 1]
            winner = _sim_knockout(t1, t2, strengths, rng)
            loser  = t2 if winner == t1 else t1
            finish[loser] = round_num
            next_round.append(winner)
        current = next_round

    if current:
        finish[current[0]] = 0  # champion

    return finish


# ── Demo groups (approximate — real groups fetched from API) ──────────────────
DEMO_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "Jamaica", "Venezuela", "Ecuador"],
    "B": ["United States", "Panama", "Bolivia", "New Zealand"],
    "C": ["Canada", "Honduras", "Guatemala", "Ivory Coast"],
    "D": ["France", "Norway", "Algeria", "South Africa"],
    "E": ["Spain", "Peru", "Ghana", "Indonesia"],
    "F": ["Germany", "Japan", "Iran", "Saudi Arabia"],
    "G": ["England", "Sweden", "Costa Rica", "Chile"],
    "H": ["Argentina", "Tunisia", "Australia", "Iceland"],
    "I": ["Brazil", "Colombia", "Egypt", "Nigeria"],
    "J": ["Portugal", "Uruguay", "Senegal", "Denmark"],
    "K": ["Netherlands", "Poland", "South Korea", "Morocco"],
    "L": ["Belgium", "Croatia", "Switzerland", "Italy"],
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(demo: bool = False) -> None:
    print(f"{'[DEMO] ' if demo else ''}WC2026 predictions starting…")

    if demo:
        matches: list[dict] = []
        groups = DEMO_GROUPS
    else:
        print("Fetching matches from football-data.org…")
        matches = fetch_wc_matches()
        # Build groups from API data
        groups_raw: dict[str, set[str]] = defaultdict(set)
        for m in matches:
            if m["stage"] == "GROUP_STAGE" and m["group"]:
                gname = m["group"].replace("GROUP_", "")
                groups_raw[gname].add(m["home"])
                groups_raw[gname].add(m["away"])
        groups = {k: sorted(v) for k, v in groups_raw.items()} if groups_raw else DEMO_GROUPS
        if not groups_raw:
            print("⚠️  No group data from API, using demo groups")

    strengths = update_strengths(matches, BASE_STRENGTH)

    # Build dict of completed group matches keyed by group name
    fixed_group: dict[str, dict[tuple, tuple]] = defaultdict(dict)
    for m in matches:
        if m["status"] != "FINISHED" or m["stage"] != "GROUP_STAGE":
            continue
        if m["home_score"] is None or m["away_score"] is None:
            continue
        gname = m["group"].replace("GROUP_", "") if m["group"] else ""
        if gname:
            fixed_group[gname][(m["home"], m["away"])] = (
                m["home_score"], m["away_score"]
            )

    matches_played   = sum(1 for m in matches if m["status"] == "FINISHED")
    total_matches    = len(matches) if matches else 104
    matches_remaining = total_matches - matches_played

    print(f"Matches played: {matches_played}/{total_matches}")
    print(f"Running {N_SIM:,} Monte Carlo simulations…")

    champion_count: dict[str, int] = defaultdict(int)
    final_count:    dict[str, int] = defaultdict(int)
    semi_count:     dict[str, int] = defaultdict(int)

    rng = np.random.default_rng()

    for _ in range(N_SIM):
        result = _simulate_one(groups, strengths, dict(fixed_group), rng)
        for team, finish in result.items():
            if finish <= 2:
                semi_count[team] += 1
            if finish <= 1:
                final_count[team] += 1
            if finish == 0:
                champion_count[team] += 1

    # Build ranking dict
    all_teams = set(BASE_STRENGTH)
    for g_teams in groups.values():
        all_teams.update(g_teams)

    # Original pre-tournament champion probability (strength-proportional)
    total_strength = sum(BASE_STRENGTH.values())
    original_p: dict[str, float] = {
        t: round(BASE_STRENGTH.get(t, 1.0) / total_strength * 100, 1)
        for t in all_teams
    }

    ranking: dict[str, dict] = {}
    for team in all_teams:
        p_champ = round(champion_count[team] / N_SIM * 100, 1)
        ranking[team] = {
            "p_champion":           p_champ,
            "p_final":              round(final_count[team]   / N_SIM * 100, 1),
            "p_semifinal":          round(semi_count[team]    / N_SIM * 100, 1),
            "p_change_vs_original": round(p_champ - original_p.get(team, 0), 1),
        }

    ranked = sorted(ranking.items(), key=lambda x: -x[1]["p_champion"])

    podium = {
        "champion":  ranked[0][0] if ranked        else "—",
        "runner_up": ranked[1][0] if len(ranked) > 1 else "—",
        "third":     ranked[2][0] if len(ranked) > 2 else "—",
    }

    movers = sorted(
        [{"team": t, "delta": v["p_change_vs_original"]} for t, v in ranking.items()],
        key=lambda x: -abs(x["delta"]),
    )[:10]

    output = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "matches_played":  matches_played,
        "matches_remaining": matches_remaining,
        "podium_current":  podium,
        "ranking":         dict(ranked),
        "next_match":      next_match_info(matches),
        "biggest_movers":  movers,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅  Saved → {OUT}")
    print(f"Champion: {podium['champion']}  "
          f"({ranking.get(podium['champion'], {}).get('p_champion', 0):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC2026 Monte Carlo predictor")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: no API calls, use base strengths")
    args = parser.parse_args()
    run(demo=args.demo)
