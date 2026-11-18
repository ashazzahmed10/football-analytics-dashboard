"""Parser tests on hand-written StatsBomb-format events (no network)."""
from app.data.statsbomb import parse_events, shot_situation


def ev(t, minute=10, **extra):
    base = {"type": {"name": t}, "period": 1, "minute": minute, "second": 30, "possession": 7,
            "team": {"id": 1}, "player": {"id": 99}, "location": [100.0, 40.0], "play_pattern": {"name": "Regular Play"}}
    base.update(extra)
    return base


def test_parse_maps_event_types():
    events = [
        ev("Pass", **{"pass": {"end_location": [110, 35], "recipient": {"id": 5}}}),
        ev("Pass", **{"pass": {"end_location": [110, 35], "outcome": {"name": "Incomplete"}, "type": {"name": "Corner"}}}),
        ev("Carry", carry={"end_location": [105, 41]}),
        ev("Shot", minute=88, shot={"statsbomb_xg": 0.31, "outcome": {"name": "Goal"}, "body_part": {"name": "Head"},
                                   "type": {"name": "Open Play"}, "end_location": [120, 40]}),
        ev("Duel", duel={"type": {"name": "Tackle"}, "outcome": {"name": "Won"}}),
        ev("Duel", duel={"type": {"name": "Tackle"}, "outcome": {"name": "Lost In Play"}}),
        ev("Interception"),
        ev("Pressure"),                                      # ignored
        ev("Shot", minute=120, period=5, shot={"statsbomb_xg": 0.7, "outcome": {"name": "Goal"},
                                               "type": {"name": "Penalty"}}),   # shoot-out: ignored
    ]
    rows, end = parse_events(42, events)
    types = [r[3] for r in rows]
    assert types == ["pass", "pass", "carry", "shot", "tackle", "interception"]
    p_ok, p_fail = rows[0], rows[1]
    assert p_ok[9] == "success" and p_ok[10] == 5 and p_fail[9] == "fail" and p_fail[13] == "corner"
    shot = rows[3]
    assert shot[9] == "goal" and shot[12] == "head" and shot[14] == 0.31 and shot[4] == 88 * 60 + 30
    assert end == 88 * 60 + 30


def test_shot_situation():
    assert shot_situation({"shot": {"type": {"name": "Penalty"}}}) == "penalty"
    assert shot_situation({"shot": {"type": {"name": "Open Play"}}, "play_pattern": {"name": "From Counter"}}) == "counter"
    assert shot_situation({"shot": {"type": {"name": "Open Play"}}, "play_pattern": {"name": "From Corner"}}) == "corner"
    assert shot_situation({"shot": {"type": {"name": "Open Play"}}, "play_pattern": {"name": "Regular Play"}}) == "open_play"
