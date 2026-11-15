FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend frontend
# Build the event database at image build time so the container starts instantly
RUN cd backend && python -c "from pathlib import Path; from app.data.league import build_database; build_database(Path('data/pitchiq.db'))"
EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
