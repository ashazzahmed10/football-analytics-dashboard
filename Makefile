.PHONY: install run test rebuild-data docker

install:
	pip install -r backend/requirements-dev.txt

run:
	cd backend && uvicorn app.main:app --reload --port 8000

test:
	cd backend && python -m pytest -q

rebuild-data:
	rm -f backend/data/pitchiq.db
	cd backend && python -c "from pathlib import Path; from app.data.league import build_database; build_database(Path('data/pitchiq.db'))"

docker:
	docker build -t pitchiq . && docker run --rm -p 8000:8000 pitchiq
