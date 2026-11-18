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


def test_websocket_replay(client):
    m = client.get("/api/matches", params={"matchday": 5}).json()[0]
    with client.websocket_connect(f"/ws/live?match_id={m['id']}&speed=60") as ws:
        first = ws.receive_json()
        assert first["type"] == "kickoff" and first["home_id"] == m["home_id"]
        last = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "fulltime":
                break
            last = msg
            p = msg["probs"]
            assert abs(p["home"] + p["draw"] + p["away"] - 1) < 1e-6
        assert last["minute"] >= 90
        assert msg["score"] == [m["home_goals"], m["away_goals"]]


def test_simulate_from_past_matchday(client):
    s = client.get("/api/simulate/season", params={"n": 500, "as_of": 6}).json()
    assert s["as_of"] == 6 and s["remaining_fixtures"] > 0
    assert abs(sum(t["title"] for t in s["teams"]) - 1) < 1e-6


def test_halfway_and_title_race(client):
    h = client.get("/api/models/halfway").json()
    assert set(h["summary"]) == {"table_ppg", "dixon_coles", "xg_poisson"}
    r = client.get("/api/simulate/title-race").json()
    for md in range(len(r["teams"][0]["series"])):
        assert abs(sum(t["series"][md]["title"] for t in r["teams"]) - 1) < 1e-6


def test_stale_service_worker_kill_switch(client):
    r = client.get("/sw.js")
    assert r.status_code == 200 and "unregister" in r.text
    assert r.headers["content-type"].startswith("application/javascript")
