"""Fetch national team squads from API-Sports and write a static JSON file.

The generated file is consumed by docs/jugadores.html. It intentionally keeps
the browser free of API keys: run this script locally or from CI with
APISPORTS_KEY configured.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


API_BASE = "https://v3.football.api-sports.io"
MAX_SQUAD_SIZE = 26

TEAMS: dict[str, dict[str, str]] = {
    "México": {"flag": "🇲🇽", "group": "A", "search": "Mexico"},
    "Jamaica": {"flag": "🇯🇲", "group": "A", "search": "Jamaica"},
    "Venezuela": {"flag": "🇻🇪", "group": "A", "search": "Venezuela"},
    "Ecuador": {"flag": "🇪🇨", "group": "A", "search": "Ecuador"},
    "United States": {"flag": "🇺🇸", "group": "B", "search": "USA"},
    "Panama": {"flag": "🇵🇦", "group": "B", "search": "Panama"},
    "Bolivia": {"flag": "🇧🇴", "group": "B", "search": "Bolivia"},
    "New Zealand": {"flag": "🇳🇿", "group": "B", "search": "New Zealand"},
    "Canada": {"flag": "🇨🇦", "group": "C", "search": "Canada"},
    "Honduras": {"flag": "🇭🇳", "group": "C", "search": "Honduras"},
    "Guatemala": {"flag": "🇬🇹", "group": "C", "search": "Guatemala"},
    "Ivory Coast": {"flag": "🇨🇮", "group": "C", "search": "Ivory Coast"},
    "France": {"flag": "🇫🇷", "group": "D", "search": "France"},
    "Norway": {"flag": "🇳🇴", "group": "D", "search": "Norway"},
    "Algeria": {"flag": "🇩🇿", "group": "D", "search": "Algeria"},
    "South Africa": {"flag": "🇿🇦", "group": "D", "search": "South Africa"},
    "Spain": {"flag": "🇪🇸", "group": "E", "search": "Spain"},
    "Peru": {"flag": "🇵🇪", "group": "E", "search": "Peru"},
    "Ghana": {"flag": "🇬🇭", "group": "E", "search": "Ghana"},
    "Indonesia": {"flag": "🇮🇩", "group": "E", "search": "Indonesia"},
    "Germany": {"flag": "🇩🇪", "group": "F", "search": "Germany"},
    "Japan": {"flag": "🇯🇵", "group": "F", "search": "Japan"},
    "Iran": {"flag": "🇮🇷", "group": "F", "search": "Iran"},
    "Saudi Arabia": {"flag": "🇸🇦", "group": "F", "search": "Saudi Arabia"},
    "England": {"flag": "🏴", "group": "G", "search": "England"},
    "Sweden": {"flag": "🇸🇪", "group": "G", "search": "Sweden"},
    "Costa Rica": {"flag": "🇨🇷", "group": "G", "search": "Costa Rica"},
    "Chile": {"flag": "🇨🇱", "group": "G", "search": "Chile"},
    "Argentina": {"flag": "🇦🇷", "group": "H", "search": "Argentina"},
    "Tunisia": {"flag": "🇹🇳", "group": "H", "search": "Tunisia"},
    "Australia": {"flag": "🇦🇺", "group": "H", "search": "Australia"},
    "Iceland": {"flag": "🇮🇸", "group": "H", "search": "Iceland"},
    "Brazil": {"flag": "🇧🇷", "group": "I", "search": "Brazil"},
    "Colombia": {"flag": "🇨🇴", "group": "I", "search": "Colombia"},
    "Egypt": {"flag": "🇪🇬", "group": "I", "search": "Egypt"},
    "Nigeria": {"flag": "🇳🇬", "group": "I", "search": "Nigeria"},
    "Portugal": {"flag": "🇵🇹", "group": "J", "search": "Portugal"},
    "Uruguay": {"flag": "🇺🇾", "group": "J", "search": "Uruguay"},
    "Senegal": {"flag": "🇸🇳", "group": "J", "search": "Senegal"},
    "Denmark": {"flag": "🇩🇰", "group": "J", "search": "Denmark"},
    "Netherlands": {"flag": "🇳🇱", "group": "K", "search": "Netherlands"},
    "Poland": {"flag": "🇵🇱", "group": "K", "search": "Poland"},
    "South Korea": {"flag": "🇰🇷", "group": "K", "search": "South Korea"},
    "Morocco": {"flag": "🇲🇦", "group": "K", "search": "Morocco"},
    "Belgium": {"flag": "🇧🇪", "group": "L", "search": "Belgium"},
    "Croatia": {"flag": "🇭🇷", "group": "L", "search": "Croatia"},
    "Switzerland": {"flag": "🇨🇭", "group": "L", "search": "Switzerland"},
    "Italy": {"flag": "🇮🇹", "group": "L", "search": "Italy"},
}

POSITION_MAP = {
    "Goalkeeper": "POR",
    "Defender": "DEF",
    "Midfielder": "MED",
    "Attacker": "DEL",
}

STAR_RATINGS = {
    "L. Messi": 91,
    "Lautaro Martínez": 87,
    "K. Mbappé": 91,
    "E. Haaland": 91,
    "V. Junior": 90,
    "H. Kane": 90,
    "J. Bellingham": 90,
    "K. De Bruyne": 88,
    "Cristiano Ronaldo": 86,
    "Neymar": 86,
    "M. Salah": 89,
}


def request_json(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(4):
        response = session.get(f"{API_BASE}/{path}", params=params, timeout=20)
        if response.status_code != 429:
            break
        wait = int(response.headers.get("Retry-After") or 20 * (attempt + 1))
        wait = max(wait, 60)
        print(f"Rate limit en {path}; espero {wait}s...")
        time.sleep(wait)
    response.raise_for_status()
    data = response.json()
    if data.get("errors"):
        errors = data["errors"]
        if isinstance(errors, dict) and "rateLimit" in errors:
            print(f"Rate limit API-Sports; espero 60s...")
            time.sleep(60)
            return request_json(session, path, params)
        raise RuntimeError(f"API-Sports error for {path}: {data['errors']}")
    return data


def find_team_id(session: requests.Session, search: str) -> int | None:
    data = request_json(session, "teams", {"search": search})
    teams = data.get("response", [])
    national = [item["team"] for item in teams if item.get("team", {}).get("national")]
    if not national:
        return None

    lowered = search.lower()
    exact = [team for team in national if team.get("name", "").lower() == lowered]
    return int((exact[0] if exact else national[0])["id"])


def estimated_rating(player: dict[str, Any]) -> int:
    name = player.get("name", "")
    if name in STAR_RATINGS:
        return STAR_RATINGS[name]

    pos = POSITION_MAP.get(player.get("position"), "DEL")
    age = int(player.get("age") or 27)
    base = {"POR": 72, "DEF": 73, "MED": 74, "DEL": 74}[pos]
    prime_bonus = max(0, 7 - abs(age - 27))
    number = player.get("number") or 30
    shirt_bonus = 3 if int(number) <= 11 else 0
    return max(62, min(84, base + prime_bonus + shirt_bonus))


def normalize_player(raw: dict[str, Any], team: str, meta: dict[str, str], idx: int) -> dict[str, Any]:
    pos = POSITION_MAP.get(raw.get("position"), "DEL")
    rating = estimated_rating(raw)
    age = int(raw.get("age") or 27)
    caps = max(0, min(130, 6 + (rating - 65) * 3 + max(0, age - 23) * 2))
    goals = 0 if pos in {"POR", "DEF"} else max(0, round((rating - 68) * (0.45 if pos == "DEL" else 0.18)))

    return {
        "id": f"{team}-{raw.get('id', idx)}",
        "apiId": raw.get("id"),
        "n": raw.get("name") or f"Jugador {idx}",
        "c": team,
        "f": meta["flag"],
        "g": meta["group"],
        "p": pos,
        "age": age,
        "club": "Selección nacional",
        "r": rating,
        "v": max(1, round((rating - 60) * (rating - 60) / 18)),
        "go": goals,
        "ca": caps,
        "wc": 1 if age >= 28 else 0,
        "photo": raw.get("photo"),
        "stickerPhoto": None,
        "number": raw.get("number"),
        "d": f"{raw.get('name') or 'Jugador'} forma parte del plantel de {team}. Foto y datos base obtenidos desde API-Sports.",
    }


def write_output(
    output: Path,
    teams_out: list[dict[str, Any]],
    missing: list[str],
    players: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "api-sports.io players/squads",
                "generatedBy": "src/fetch_squads.py",
                "teams": teams_out,
                "missing": missing,
                "players": players,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fetch_squads(output: Path, pause: float) -> None:
    load_dotenv()
    api_key = os.getenv("APISPORTS_KEY")
    if not api_key:
        raise SystemExit("Missing APISPORTS_KEY. Set it in the environment or .env.")

    session = requests.Session()
    session.headers.update({"x-apisports-key": api_key})

    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        players = current.get("players", [])
        teams_out = current.get("teams", [])
        missing = current.get("missing", [])
    else:
        players = []
        teams_out = []
        missing = []

    completed = {team["name"] for team in teams_out}

    for team_name, meta in TEAMS.items():
        if team_name in completed:
            print(f"{team_name}: ya estaba descargado")
            continue

        team_id = find_team_id(session, meta["search"])
        if team_id is None:
            missing.append(team_name)
            write_output(output, teams_out, missing, players)
            continue

        data = request_json(session, "players/squads", {"team": team_id})
        response = data.get("response", [])
        squad = (response[0].get("players", []) if response else [])[:MAX_SQUAD_SIZE]
        teams_out.append({"name": team_name, "id": team_id, "count": len(squad), **meta})

        for idx, raw in enumerate(squad, start=1):
            players.append(normalize_player(raw, team_name, meta, idx))

        print(f"{team_name}: {len(squad)} jugadores")
        write_output(output, teams_out, missing, players)
        time.sleep(pause)

    write_output(output, teams_out, missing, players)
    print(f"Wrote {len(players)} players to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/squads.json", type=Path)
    parser.add_argument("--pause", default=0.2, type=float)
    args = parser.parse_args()
    fetch_squads(args.output, args.pause)


if __name__ == "__main__":
    main()
