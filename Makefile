.PHONY: install run run-synthetic test rebuild-data docker

install:
	pip install -r backend/requirements-dev.txt

run:
	cd backend && python -m uvicorn app.main:app --reload --port 8000

run-synthetic:
	cd backend && PITCHIQ_SOURCE=synthetic python -m uvicorn app.main:app --reload --port 8000

test:
	cd backend && python -m pytest -q

rebuild-data:   # re-download Premier League 2015/16 from StatsBomb open data
	rm -f backend/data/statsbomb.db
	cd backend && python -m app.data.statsbomb --gzip

docker:
	docker build -t pitchiq . && docker run --rm -p 8000:8000 pitchiq
