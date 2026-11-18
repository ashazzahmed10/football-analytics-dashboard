"""
StatsBomb open-data importer.

Downloads a full league season from https://github.com/statsbomb/open-data
(free for non-commercial use - please credit StatsBomb), keeps only the event
types PitchIQ models, and writes them to the same SQLite schema the rest of the
platform reads. Events are streamed match-by-match, so the ~1.2 GB of raw JSON is
never held on disk or in memory at once.

    python -m app.data.statsbomb                         # Premier League 2015/16 (default)
    python -m app.data.statsbomb --competition 11 --season 27 --out data/laliga1516.db
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .league import SCHEMA

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SEASONS = {  # full-coverage league seasons in the open data
    (2, 27): "Premier League 2015/16", (11, 27): "La Liga 2015/16",
    (12, 27): "Serie A 2015/16", (7, 27): "Ligue 1 2015/16",
}
SHORT = {
    "AFC Bournemouth": "BOU", "Arsenal": "ARS", "Aston Villa": "AVL", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Leicester City": "LEI", "Liverpool": "LIV",
    "Manchester City": "MCI", "Manchester United": "MUN", "Newcastle United": "NEW",
    "Norwich City": "NOR", "Southampton": "SOU", "Stoke City": "STK", "Sunderland": "SUN",
    "Swansea City": "SWA", "Tottenham Hotspur": "TOT", "Watford": "WAT",
    "West Bromwich Albion": "WBA", "West Ham United": "WHU",
    "Barcelona": "BAR", "Real Madrid": "RMA", "Atlético Madrid": "ATM", "Juventus": "JUV",
    "Paris Saint-Germain": "PSG",
}
POSITION = {
    "Goalkeeper": "GK", "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Right Back": "RB", "Right Wing Back": "RB", "Left Back": "LB", "Left Wing Back": "LB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "CM", "Left Attacking Midfield": "LW", "Right Attacking Midfield": "RW",
    "Right Midfield": "RW", "Right Wing": "RW", "Left Midfield": "LW", "Left Wing": "LW",
    "Center Forward": "ST", "Left Center Forward": "ST", "Right Center Forward": "ST", "Secondary Striker": "ST",
}
SHOT_OUTCOME = {"Goal": "goal", "Saved": "saved", "Saved to Post": "saved", "Saved Off Target": "saved",
                "Blocked": "blocked", "Off T": "off_target", "Post": "off_target", "Wayward": "off_target"}
TACKLE_WON = {"Won", "Success", "Success In Play", "Success Out"}


def fetch_json(path: str, retries: int = 4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{path}", timeout=60) as r:
                return json.load(r)
        except Exception:  # network hiccup - back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _clock(mmss: Optional[str], default: int) -> int:
    if not mmss:
        return default
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def shot_situation(ev: dict) -> str:
    st = ev["shot"].get("type", {}).get("name")
    if st == "Penalty":
        return "penalty"
    if st == "Free Kick":
        return "free_kick"
    if st == "Corner":
        return "corner"
    pattern = ev.get("play_pattern", {}).get("name")
    if pattern == "From Counter":
        return "counter"
    if pattern == "From Corner":
        return "corner"
    return "open_play"


def parse_events(match_id: int, events: list[dict]) -> tuple[list[tuple], int]:
    """Map raw StatsBomb events to PitchIQ event rows. Returns (rows, match_end_second)."""
    rows = []
    end = 0
    for ev in events:
        if ev.get("period", 1) > 4:          # ignore penalty shoot-outs
            continue
        sec = ev["minute"] * 60 + ev["second"]
        end = max(end, sec)
        t = ev["type"]["name"]
        loc = ev.get("location")
        if not loc or "player" not in ev:
            continue
        base = dict(match_id=match_id, team_id=ev["team"]["id"], player_id=ev["player"]["id"], second=sec,
                    x=loc[0], y=loc[1], possession=ev.get("possession"))
        if t == "Pass":
            p = ev["pass"]
            ptype = p.get("type", {}).get("name")
            rows.append(dict(base, type="pass", end_x=p["end_location"][0], end_y=p["end_location"][1],
                             outcome="fail" if "outcome" in p else "success",
                             receiver_id=p.get("recipient", {}).get("id"),
                             situation={"Corner": "corner", "Free Kick": "free_kick", "Goal Kick": "goal_kick",
                                        "Kick Off": "kickoff", "Throw-in": "throw_in"}.get(ptype)))
        elif t == "Carry":
            c = ev["carry"]
            rows.append(dict(base, type="carry", end_x=c["end_location"][0], end_y=c["end_location"][1],
                             outcome="success"))
        elif t == "Dribble":
            ok = ev.get("dribble", {}).get("outcome", {}).get("name") == "Complete"
            rows.append(dict(base, type="dribble", outcome="success" if ok else "fail"))
        elif t == "Shot":
            s = ev["shot"]
            el = s.get("end_location", [120, 40])
            rows.append(dict(base, type="shot", end_x=el[0], end_y=el[1],
                             outcome=SHOT_OUTCOME.get(s.get("outcome", {}).get("name"), "off_target"),
                             body_part="head" if s.get("body_part", {}).get("name") == "Head" else "foot",
                             situation=shot_situation(ev), true_xg=s.get("statsbomb_xg")))
        elif t == "Interception":
            rows.append(dict(base, type="interception"))
        elif t == "Duel" and ev.get("duel", {}).get("type", {}).get("name") == "Tackle" \
                and ev["duel"].get("outcome", {}).get("name") in TACKLE_WON:
            rows.append(dict(base, type="tackle"))
    cols = ("match_id", "team_id", "player_id", "type", "second", "x", "y", "end_x", "end_y", "outcome",
            "receiver_id", "possession", "body_part", "situation", "true_xg")
    return [tuple(r.get(c) for c in cols) for r in rows], end


def build_statsbomb_database(out: Path, competition: int = 2, season: int = 27, workers: int = 8) -> None:
    t0 = time.time()
    name = SEASONS.get((competition, season), f"competition {competition} / season {season}")
    print(f"[pitchiq] downloading StatsBomb open data: {name}")
    matches = fetch_json(f"matches/{competition}/{season}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)

    teams = {}
    for m in matches:
        for side in ("home", "away"):
            t = m[f"{side}_team"]
            teams[t[f"{side}_team_id"]] = t[f"{side}_team_name"]
    con.executemany("INSERT INTO teams (id, name, short) VALUES (?,?,?)",
                    [(tid, nm, SHORT.get(nm, nm.replace("FC ", "")[:3].upper())) for tid, nm in teams.items()])
    con.executemany("INSERT INTO matches (id, matchday, home_id, away_id, home_goals, away_goals, played, date) "
                    "VALUES (?,?,?,?,?,?,1,?)",
                    [(m["match_id"], m["match_week"], m["home_team"]["home_team_id"], m["away_team"]["away_team_id"],
                      m["home_score"], m["away_score"], m["match_date"]) for m in matches])

    players: dict[int, dict] = {}
    pos_minutes: dict[int, Counter] = defaultdict(Counter)
    team_minutes: dict[int, Counter] = defaultdict(Counter)

    def work(m):
        mid = m["match_id"]
        return mid, fetch_json(f"events/{mid}.json"), fetch_json(f"lineups/{mid}.json")

    done, n_events = 0, 0
    with ThreadPoolExecutor(workers) as pool:
        for mid, events, lineups in pool.map(work, matches):
            rows, end = parse_events(mid, events)
            con.executemany("INSERT INTO events (match_id,team_id,player_id,type,second,x,y,end_x,end_y,outcome,"
                            "receiver_id,possession,body_part,situation,true_xg) VALUES "
                            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            n_events += len(rows)
            apps = []
            for team in lineups:
                for pl in team["lineup"]:
                    pid = pl["player_id"]
                    players.setdefault(pid, {"name": pl.get("player_nickname") or pl["player_name"],
                                             "shirt": pl.get("jersey_number"),
                                             "country": (pl.get("country") or {}).get("name")})
                    mins = 0
                    for p in pl.get("positions", []):
                        a, b = _clock(p.get("from"), 0), _clock(p.get("to"), end)
                        dur = max(0, b - a) // 60
                        mins += dur
                        pos_minutes[pid][POSITION.get(p["position"], "CM")] += dur
                    if mins > 0:
                        apps.append((mid, pid, team["team_id"], mins))
                        team_minutes[pid][team["team_id"]] += mins
            con.executemany("INSERT INTO appearances VALUES (?,?,?,?)", apps)
            done += 1
            if done % 40 == 0 or done == len(matches):
                print(f"[pitchiq]   {done}/{len(matches)} matches · {n_events:,} events · {time.time() - t0:.0f}s")

    rows = []
    for pid, info in players.items():
        if not team_minutes[pid]:
            continue
        rows.append((pid, team_minutes[pid].most_common(1)[0][0], info["name"],
                     pos_minutes[pid].most_common(1)[0][0], info["shirt"], info["country"]))
    con.executemany("INSERT INTO players (id, team_id, name, position, shirt, country) VALUES (?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("source", "statsbomb"), ("competition", name),
        ("attribution", "Event data: StatsBomb open data (github.com/statsbomb/open-data)")])
    con.commit()
    con.execute("VACUUM")
    con.close()
    shutil.move(tmp, out)
    print(f"[pitchiq] built {out.name}: {len(matches)} matches, {len(rows)} players, "
          f"{n_events:,} events in {time.time() - t0:.0f}s")


def compress(db: Path) -> Path:
    gz = db.with_suffix(db.suffix + ".gz")
    with open(db, "rb") as f, gzip.open(gz, "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    return gz


def decompress(gz: Path, db: Path) -> None:
    with gzip.open(gz, "rb") as g, open(db, "wb") as f:
        shutil.copyfileobj(g, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition", type=int, default=2)
    ap.add_argument("--season", type=int, default=27)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "statsbomb.db")
    ap.add_argument("--gzip", action="store_true", help="also write a .gz copy for committing")
    a = ap.parse_args()
    build_statsbomb_database(a.out, a.competition, a.season)
    if a.gzip:
        print(f"[pitchiq] compressed -> {compress(a.out)}")
