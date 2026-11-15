def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "X-Process-Time-ms" in r.headers


def test_table_is_sorted(client):
    t = client.get("/api/league/table").json()
    assert len(t) == 20
    assert [r["pts"] for r in t] == sorted([r["pts"] for r in t], reverse=True)


def test_match_detail(client):
    ms = client.get("/api/matches", params={"matchday": 1}).json()
    d = client.get(f"/api/matches/{ms[0]['id']}").json()
    assert d["home"]["goals"] == ms[0]["home_goals"]
    assert {"nodes", "edges", "metrics"} <= set(d["home"]["network"])
    assert abs(sum(d["deserved"].values()) - 1) < 1e-6


def test_unplayed_match_is_409(client):
    ms = client.get("/api/matches", params={"matchday": 38}).json()
    assert client.get(f"/api/matches/{ms[0]['id']}").status_code == 409


def test_xg_predict(client):
    r = client.post("/api/models/xg/predict", json=[{"x": 114, "y": 40}, {"x": 90, "y": 40},
                                                    {"x": 108, "y": 40, "situation": "penalty"}])
    a, b, pen = [s["xg"] for s in r.json()]
    assert a > b and 0.6 < pen < 0.9
    assert client.post("/api/models/xg/predict", json=[{"x": 200, "y": 40}]).status_code == 422


def test_predict_and_simulate(client):
    p = client.get("/api/predict", params={"home": 1, "away": 2}).json()
    for m in p["models"].values():
        assert abs(sum(m.values()) - 1) < 1e-6
    s = client.get("/api/simulate/season", params={"n": 500}).json()
    assert abs(sum(t["title"] for t in s["teams"]) - 1) < 1e-6
    assert client.get("/api/predict", params={"home": 1, "away": 1}).status_code == 422


def test_players(client):
    m = client.get("/api/players/map").json()
    pid = m["players"][0]["player_id"]
    d = client.get(f"/api/players/{pid}").json()
    assert len(d["similar"]) == 8 and "percentiles" in d["profile"]


def test_websocket_live(client):
    with client.websocket_connect("/ws/live?home=1&away=2&speed=60&seed=1") as ws:
        first = ws.receive_json()
        assert first["type"] == "kickoff"
        last = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "fulltime":
                break
            last = msg
            p = msg["probs"]
            assert abs(p["home"] + p["draw"] + p["away"] - 1) < 1e-6
        assert last["minute"] >= 90
