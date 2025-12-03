# Rugby Vision Backend

## Overview

FastAPI backend service for Rugby Vision - multi-camera 3D forward pass detection system.

## Setup

### Prerequisites

- Python 3.11+
- pip or poetry

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using poetry
poetry install
```

## Development

### Run the server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python main.py
```

### Run tests

```bash
pytest

# With coverage
pytest --cov=. --cov-report=term-missing
```

### Code quality

```bash
# Format code
black .

# Lint
flake8 .

# Type check
mypy .
```

## API Endpoints

### POST /api/clip/analyse-pass

Analyse a pass from multi-camera video.

**Request:**
```json
{
  "clip_id": "demo-clip-1",
  "cameras": ["cam1", "cam2", "cam3"],
  "start_time": 0.0,
  "end_time": 5.0
}
```

**Response:**
```json
{
  "is_forward": false,
  "confidence": 0.85,
  "explanation": "Pass deemed legal based on 3D trajectory analysis",
  "metadata": {
    "clip_id": "demo-clip-1",
    "cameras_used": ["cam1", "cam2", "cam3"],
    "duration_seconds": 5.0,
    "phase": "POC"
  }
}
```

### GET /api/clip/{clip_id}/debug-data

Get debug data for a specific clip.

### GET /health

Health check endpoint.

## Architecture

See `../ARCHITECTURE_OVERVIEW.md` for full system architecture.

## Coding Standards

See `../CONTRIBUTING.md` for coding rules and guidelines.
