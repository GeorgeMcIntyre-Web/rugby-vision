# Rugby Vision 🏉

Multi-camera 3D forward pass detection system for rugby using computer vision and physics-based analysis.

## Overview

Rugby Vision analyzes video from multiple camera angles to determine whether a pass is forward according to rugby laws. The system reconstructs 3D positions of players and the ball, then applies physics-based criteria to make accurate decisions with confidence scores.

**Target Users:** Television Match Officials (TMOs), referees, and analysts

## Features

- **Multi-Camera Analysis**: Synchronizes and processes video from multiple angles
- **3D Reconstruction**: Computes real-world 3D positions from 2D camera views
- **Physics-Based Decisions**: Applies rigorous criteria considering ball trajectory and player momentum
- **Confidence Scoring**: Provides transparency with confidence percentages
- **Clean UI**: Simple, referee-friendly interface for match officials

## Project Status: Phase 1 - Proof of Concept

Current phase focuses on offline analysis of recorded clips with synthetic/mock data.

**Roadmap:**
- **Phase 1** (Current): Offline POC with recorded clips
- **Phase 2**: Semi-real-time analysis with small delays
- **Phase 3**: Fully integrated live system

See `PLAN_RUGBY_VISION.md` for complete roadmap.

## Architecture

```
Frontend (React + TypeScript)
    ↓ REST API
Backend (Python + FastAPI)
    ↓
ML/CV Pipeline:
  1. Video Ingestion & Sync
  2. Detection & Tracking
  3. 3D Reconstruction
  4. Decision Engine
```

See `ARCHITECTURE_OVERVIEW.md` for detailed architecture documentation.

## Quick Start

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.11+ (for backend)
- **Docker** (optional, for containerized deployment)

### Local Development Setup

#### 1. Clone the repository

```bash
git clone <repository-url>
cd rugby-vision
```

#### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

#### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

#### 4. Using Docker (Alternative)

```bash
# From project root
docker-compose up --build
```

This starts both frontend and backend services:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Project Structure

```
rugby-vision/
├── frontend/              # React + TypeScript UI
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API client
│   │   ├── App.tsx        # Main app component
│   │   └── main.tsx       # Entry point
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/               # Python FastAPI service
│   ├── api/               # API endpoints
│   ├── tests/             # Backend tests
│   ├── main.py            # FastAPI app entry point
│   ├── video_ingest.py    # Video loading & sync
│   ├── video_sync.py      # Frame synchronization
│   ├── requirements.txt
│   └── pyproject.toml
│
├── ml/                    # ML/CV components
│   ├── models/            # Trained models
│   ├── training/          # Training scripts
│   ├── tests/             # ML tests
│   └── mock_data_generator.py
│
├── infra/                 # Infrastructure & CI/CD
│   ├── ci/                # CI/CD configs
│   └── docker/            # Docker configs
│
├── ARCHITECTURE_OVERVIEW.md
├── CONTRIBUTING.md
├── RUGBY_VISION_REQUIREMENTS.md
├── VIDEO_INGEST_DESIGN.md
├── PLAN_RUGBY_VISION.md
├── docker-compose.yml
└── README.md
```

## Development Workflow

### Running Tests

**Backend:**
```bash
cd backend
pytest
pytest --cov=. --cov-report=term-missing  # With coverage
```

**Frontend:**
```bash
cd frontend
npm test
```

### Code Quality

**Backend:**
```bash
# Format
black .

# Lint
flake8 .

# Type check
mypy .
```

**Frontend:**
```bash
# Lint
npm run lint

# Type check
npm run type-check
```

### Coding Standards

This project follows strict coding rules:
- ✅ Use guard clauses (avoid deep nesting)
- ✅ Avoid `else` blocks when possible
- ✅ Maximum 2 levels of nesting
- ✅ Explicit types everywhere (no `any`, full type hints)
- ✅ Compact, readable functions

See `CONTRIBUTING.md` for complete guidelines.

## API Endpoints

### POST `/api/clip/analyse-pass`

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
    "duration_seconds": 5.0
  }
}
```

### GET `/api/clip/{clip_id}/debug-data`

Get debug data for a specific clip (for analysts).

### GET `/health`

Health check endpoint.

## Documentation

- **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**: System architecture and design
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Coding standards and contribution guidelines
- **[RUGBY_VISION_REQUIREMENTS.md](RUGBY_VISION_REQUIREMENTS.md)**: Detailed requirements and specifications
- **[VIDEO_INGEST_DESIGN.md](VIDEO_INGEST_DESIGN.md)**: Video processing pipeline design
- **[PLAN_RUGBY_VISION.md](PLAN_RUGBY_VISION.md)**: Full project roadmap (phases 1-13)

## Technology Stack

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- Axios (HTTP client)

**Backend:**
- Python 3.11+
- FastAPI (web framework)
- Pydantic v2 (validation)
- OpenCV (video processing)
- NumPy (numerical operations)

**Infrastructure:**
- Docker + Docker Compose
- GitHub Actions (CI/CD)

## Contributing

We welcome contributions! Please:

1. Read `CONTRIBUTING.md` for coding standards
2. Follow the guard clause and type safety rules
3. Write tests for new features
4. Update documentation as needed

## License

[To be determined]

## Contact

[To be added]

## Acknowledgments

Built for improving the accuracy and transparency of rugby officiating decisions.
