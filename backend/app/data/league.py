"""
Builds a fictional 20-team league, simulates the played part of the season with the
match engine, and persists everything to SQLite.
"""
from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path

from .engine import FORMATION, MatchSimulator, PlayerState, TeamState

TEAMS = [
    ("Northbridge Athletic", "NBA"), ("Harbour City", "HBC"), ("Redmoor United", "RMU"),
    ("Kingsport Rovers", "KPR"), ("Eastvale Town", "EVT"), ("Silverlake FC", "SLK"),
    ("Ironbridge Wanderers", "IBW"), ("Ashford Rangers", "ASR"), ("Brookhaven Albion", "BHA"),
    ("Westmere County", "WMC"), ("Oakridge Forest", "OKF"), ("Stonegate City", "SGC"),
    ("Millbrook Villa", "MBV"), ("Falconcrest FC", "FCC"), ("Rivertown Orient", "RTO"),
    ("Thornbury Park", "TBP"), ("Clearwater Dynamo", "CWD"), ("Highfield Rovers", "HFR"),
    ("Saltmarsh United", "SMU"), ("Greystone Athletic", "GSA"),
]
FIRST = ["Leo", "Marco", "Kai", "Tomas", "Rafael", "Noah", "Elias", "Luca", "Mateo", "Arjun", "Yusuf",
         "Ivan", "Hugo", "Sami", "Diego", "Felix", "Kenji", "Omar", "Andre", "Jonas", "Theo", "Nico",
         "Ravi", "Emil", "Pablo", "Dani", "Oscar", "Milan", "Adam", "Bruno", "Caleb", "Dario", "Ezra",
         "Finn", "Gabriel", "Isak", "Jude", "Karim", "Lars", "Musa", "Nils", "Otto", "Pedro", "Quinn",
         "Rui", "Stefan", "Tariq", "Umar", "Viktor", "Wes", "Xavi", "Yann", "Zane", "Aiden", "Bastian"]
LAST = ["Moreno", "Okafor", "Lindqvist", "Varga", "Sato", "Mensah", "Kowalski", "Duarte", "Haddad",
        "Novak", "Petrov", "Rossi", "Fischer", "Bakker", "Silva", "Nkemelu", "Jansen", "Costa", "Ortega",
        "Brandt", "Yilmaz", "Dubois", "Kaur", "Reyes", "Holm", "Marin", "Adeyemi", "Kovac", "Laine",
        "Sousa", "Weber", "Ibrahim", "Castillo", "Moretti", "Eriksen", "Banda", "Ferreira", "Tanaka",
        "Kruger", "Vidal", "Osei", "Lund", "Romero", "Nakamura", "Hassan", "Albers", "Quaresma", "Ivanov"]

ROLE_PROFILE = {   # (passing, finishing, dribbling, defending) role offsets
    "GK": (-0.5, -2.0, -1.5, 1.0), "CB": (-0.2, -1.2, -0.8, 1.0), "RB": (0.1, -0.8, 0.3, 0.5),
    "LB": (0.1, -0.8, 0.3, 0.5), "DM": (0.5, -0.6, -0.1, 0.7), "CM": (0.6, -0.2, 0.2, 0.2),
    "RW": (0.2, 0.5, 0.9, -0.6), "LW": (0.2, 0.5, 0.9, -0.6), "ST": (-0.1, 1.0, 0.4, -0.8),
}
PLAYED_MATCHDAYS = 26
SCHEMA = """
CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT, short TEXT,
    true_attack REAL, true_defence REAL, directness REAL);
CREATE TABLE players (id INTEGER PRIMARY KEY, team_id INTEGER, name TEXT, position TEXT,
    shirt INTEGER, age INTEGER, starter INTEGER, ax REAL, ay REAL,
    passing REAL, finishing REAL, dribbling REAL, defending REAL);
CREATE TABLE matches (id INTEGER PRIMARY KEY, matchday INTEGER, home_id INTEGER, away_id INTEGER,
    home_goals INTEGER, away_goals INTEGER, played INTEGER, seed INTEGER);
CREATE TABLE appearances (match_id INTEGER, player_id INTEGER, team_id INTEGER, minutes INTEGER);
CREATE TABLE events (id INTEGER PRIMARY KEY, match_id INTEGER, team_id INTEGER, player_id INTEGER,
    type TEXT, second INTEGER, x REAL, y REAL, end_x REAL, end_y REAL, outcome TEXT,
    receiver_id INTEGER, possession INTEGER, body_part TEXT, situation TEXT, true_xg REAL);
CREATE INDEX ev_match ON events(match_id);
CREATE INDEX ev_type ON events(type);
CREATE INDEX ev_player ON events(player_id);
"""


def round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Circle-method double round robin -> list of matchdays of (home, away)."""
    ids = list(range(n))
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = ids[i], ids[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def build_teams(rng: random.Random) -> tuple[list[dict], list[dict]]:
    teams, players = [], []
    used = set()
    pid = 1
    for tid, (name, short) in enumerate(TEAMS, start=1):
        quality = rng.gauss(0, 0.22)
        t = dict(id=tid, name=name, short=short,
                 true_attack=round(quality + rng.gauss(0, 0.2), 3),
                 true_defence=round(quality + rng.gauss(0, 0.2), 3),
                 directness=round(rng.uniform(-0.6, 0.9), 3))
        teams.append(t)
        roster = FORMATION + [("GK", 8, 40), ("CB", 30, 40), ("CM", 62, 40), ("RW", 88, 68), ("ST", 98, 40)]
        for i, (pos, ax, ay) in enumerate(roster):
            while True:
                nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if nm not in used:
                    used.add(nm)
                    break
            prof = ROLE_PROFILE[pos]
            starter = i < 11
            lvl = quality * 0.6 + (0 if starter else -0.35)
            players.append(dict(
                id=pid, team_id=tid, name=nm, position=pos, shirt=i + 1 if i else 1,
                age=rng.randint(18, 34), starter=int(starter), ax=ax, ay=ay,
                passing=round(prof[0] + lvl + rng.gauss(0, 0.45), 3),
                finishing=round(prof[1] + lvl + rng.gauss(0, 0.45), 3),
                dribbling=round(prof[2] + lvl + rng.gauss(0, 0.45), 3),
                defending=round(prof[3] + lvl + rng.gauss(0, 0.45), 3)))
            pid += 1
    return teams, players


def team_state(team: dict, squad: list[dict], rng: random.Random) -> TeamState:
    """Pick a matchday XI (light rotation) in formation order."""
    starters = [p for p in squad if p["starter"]]
    bench = [p for p in squad if not p["starter"]]
    lineup = []
    for p in starters:
        sub = next((b for b in bench if b["position"] == p["position"] or
                    (p["position"] in ("LW", "RW") and b["position"] in ("LW", "RW"))), None)
        chosen = sub if (sub and rng.random() < 0.12 and sub not in lineup) else p
        lineup.append(chosen)
    ts = TeamState(team["id"], team["true_attack"], team["true_defence"], team["directness"])
    for slot, p in zip(FORMATION, lineup):
        ts.lineup.append(PlayerState(p["id"], p["name"], slot[0], slot[1], slot[2],
                                     p["passing"], p["finishing"], p["dribbling"], p["defending"]))
    return ts


def build_database(path: Path, seed: int = 11, played_matchdays: int = PLAYED_MATCHDAYS) -> None:
    t0 = time.time()
    rng = random.Random(seed)
    teams, players = build_teams(rng)
    squads = {t["id"]: [p for p in players if p["team_id"] == t["id"]] for t in teams}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO teams VALUES (:id,:name,:short,:true_attack,:true_defence,:directness)", teams)
    con.executemany("INSERT INTO players VALUES (:id,:team_id,:name,:position,:shirt,:age,:starter,:ax,:ay,"
                    ":passing,:finishing,:dribbling,:defending)", players)

    schedule = round_robin(len(teams))
    mid = 0
    ev_rows = []
    for md, fixtures in enumerate(schedule, start=1):
        for h, a in fixtures:
            mid += 1
            home, away = teams[h], teams[a]
            mseed = rng.randrange(1 << 30)
            if md > played_matchdays:
                con.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?)",
                            (mid, md, home["id"], away["id"], None, None, 0, mseed))
                continue
            mrng = random.Random(mseed)
            hs, as_ = team_state(home, squads[home["id"]], mrng), team_state(away, squads[away["id"]], mrng)
            sim = MatchSimulator(hs, as_, mrng)
            for e in sim.run():
                ev_rows.append((mid, e.team_id, e.player_id, e.type, e.second, round(e.x, 2), round(e.y, 2),
                                None if e.end_x is None else round(e.end_x, 2),
                                None if e.end_y is None else round(e.end_y, 2), e.outcome, e.receiver_id,
                                e.possession, e.body_part, e.situation, e.true_xg))
            con.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?)",
                        (mid, md, home["id"], away["id"], sim.score[hs.id], sim.score[as_.id], 1, mseed))
            mins = sim.end_second // 60
            con.executemany("INSERT INTO appearances VALUES (?,?,?,?)",
                            [(mid, p.id, ts.id, mins) for ts in (hs, as_) for p in ts.lineup])
    con.executemany("INSERT INTO events (match_id,team_id,player_id,type,second,x,y,end_x,end_y,outcome,"
                    "receiver_id,possession,body_part,situation,true_xg) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ev_rows)
    con.commit()
    con.close()
    print(f"[pitchiq] built {path.name}: {mid} fixtures, {len(ev_rows):,} events in {time.time() - t0:.1f}s")


def load_team_state(con: sqlite3.Connection, team_id: int, rng: random.Random) -> TeamState:
    con.row_factory = sqlite3.Row
    team = dict(con.execute("SELECT * FROM teams WHERE id=?", (team_id,)).fetchone())
    squad = [dict(r) for r in con.execute("SELECT * FROM players WHERE team_id=?", (team_id,))]
    return team_state(team, squad, rng)
