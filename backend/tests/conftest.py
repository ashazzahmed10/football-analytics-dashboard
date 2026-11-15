import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def small_db(tmp_path_factory):
    from app.data.league import build_database
    db = tmp_path_factory.mktemp("data") / "test.db"
    build_database(db, seed=5, played_matchdays=12)
    return db


@pytest.fixture(scope="session")
def client(small_db):
    os.environ["PITCHIQ_DB"] = str(small_db)
    import importlib
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
