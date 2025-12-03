# Rugby Vision - Mega Prompt for Multi-Agent Development

## 📖 How to Use This Document

**Workflow**: 
1. Your human will tell you which **agent type** or **specific task** you are assigned
2. Use the Quick Navigation section below to jump to your assigned section
3. Read your task details, specifications, and requirements
4. Review the Coding Standards section (CRITICAL - must follow)
5. Implement your assigned work following the specifications
6. Write tests as required
7. Follow the Git Workflow to commit your changes

**Example**: If you're told "You are a Backend Agent working on Phase 7", jump to [Phase 7: Backend API](#-phase-7-backend-api-and-pipeline-glue) and follow those instructions.

---

## 🚀 Quick Navigation for Agents

**How to use this document**: Your human will tell you which agent type/task you are. Use the links below to jump directly to your section.

### Agent Type Quick Links

**Backend/ML Agents** (Phases 6-7):
- [Phase 6: Decision Engine](#-phase-6-forward-pass-decision-engine) - Start here for Phase 6
- [Phase 7: Backend API](#-phase-7-backend-api-and-pipeline-glue) - Start here for Phase 7
- [Quick Start: Backend/ML](#-quick-start-for-agents)

**Frontend Agents** (Phase 8):
- [Phase 8: Frontend UI](#-phase-8-frontend-ui-for-referees-and-analysts) - Start here
- [Quick Start: Frontend](#-quick-start-for-agents)

**Testing Agents** (Phase 9):
- [Phase 9: Testing Strategy](#-phase-9-testing-strategy) - Start here
- [Quick Start: Testing](#-quick-start-for-agents)

**DevOps Agents** (Phase 10):
- [Phase 10: CI/CD](#-phase-10-cicd-and-deployment-pipeline) - Start here

**ML/Data Agents** (Phase 11):
- [Phase 11: Datasets and Training](#-phase-11-datasets-labeling-and-model-training-parallel) - Start here

**Performance Agents** (Phase 12):
- [Phase 12: Performance Optimization](#-phase-12-performance-optimization) - Start here

**Documentation Agents** (Phase 13):
- [Phase 13: Documentation and Demo](#-phase-13-documentation-and-demo) - Start here

### Task-Specific Quick Links

**Phase 6 Tasks**:
- [Task 6.1: Decision Criteria](#task-61-decision-criteria-definition) - Physics/ML Agent
- [Task 6.2: Pass Event Detection](#task-62-pass-event-detection) - ML/CV Agent
- [Task 6.3: Ball Trajectory Analysis](#task-63-ball-trajectory-analysis) - ML/CV Agent
- [Task 6.4: Decision Logic](#task-64-decision-logic-implementation) - Core ML Agent
- [Task 6.5: Confidence Scoring](#task-65-confidence-scoring) - ML Agent
- [Task 6.6: Testing](#task-66-testing-and-validation) - Testing Agent

---

## 🎯 Project Overview

**Rugby Vision** is a multi-camera 3D forward pass detection system for rugby officials. The system processes synchronized video from multiple camera angles to reconstruct 3D positions of players and the ball, then determines whether a pass is forward according to rugby laws.

**Repository**: `https://github.com/GeorgeMcIntyre-Web/rugby-vision`  
**Current Status**: Phases 1-5 COMPLETE ✅ | Phase 6+ IN PROGRESS

---

## 📊 Current Project Status

### ✅ COMPLETED (Phases 1-5)

**Phase 1**: Project Bootstrap
- Monorepo structure (`/frontend`, `/backend`, `/ml`, `/infra`)
- React + TypeScript frontend scaffolding
- Python FastAPI backend scaffolding
- Documentation (ARCHITECTURE_OVERVIEW.md, CONTRIBUTING.md)

**Phase 2**: Requirements and Use Case Spec
- RUGBY_VISION_REQUIREMENTS.md
- User workflows documented
- Success metrics defined

**Phase 3**: Multi-Camera Ingestion and Synchronization
- `backend/video_ingest.py` - VideoIngestor class
- `backend/video_sync.py` - VideoSynchronizer class
- `ml/mock_data_generator.py` - Synthetic video generator
- Unit tests in `backend/tests/`

**Phase 4**: Player and Ball Detection and Tracking ✅
- `ml/detector.py` - Object detection (mock/stub, ready for YOLO)
- `ml/tracker.py` - IOU-based tracking
- `ml/detection_tracking_api.py` - Orchestration API
- Backend endpoint: `POST /api/clip/detect-and-track`
- Comprehensive tests in `ml/tests/`

**Phase 5**: 3D Reconstruction ✅
- `ml/calibration.py` - Camera calibration handling
- `ml/triangulation.py` - Multi-view triangulation
- `ml/spatial_model.py` - Field coordinate system
- `ml/field_coords.py` - Field geometry utilities
- Tests for all geometry functions

### 🚧 IN PROGRESS / TODO (Phases 6-13)

**Phase 6**: Forward Pass Decision Engine ⚠️ **NEXT PRIORITY**
**Phase 7**: Backend API and Pipeline Glue
**Phase 8**: Frontend UI for Referees and Analysts
**Phase 9**: Testing Strategy
**Phase 10**: CI/CD and Deployment Pipeline
**Phase 11**: Datasets, Labeling, and Model Training (can run in parallel)
**Phase 12**: Performance, Latency, and Real-Time
**Phase 13**: Documentation, Demo, and Pitch Material

---

## 🏗️ Architecture Overview

```
Frontend (React + TypeScript)
    ↓ REST API
Backend (Python + FastAPI)
    ↓
ML/CV Pipeline:
  1. Video Ingestion & Sync ✅
  2. Detection & Tracking ✅
  3. 3D Reconstruction ✅
  4. Decision Engine ⚠️ TODO
```

### Technology Stack

**Frontend**: React 18+ with TypeScript, Vite, Axios  
**Backend**: FastAPI (Python 3.11+), Pydantic v2, Uvicorn  
**ML/CV**: OpenCV, NumPy, (Future: PyTorch, YOLO)  
**Infrastructure**: Docker, Docker Compose, GitHub Actions

---

## 📋 CODING STANDARDS (CRITICAL - MUST FOLLOW)

All agents MUST adhere to these strict coding rules:

### ✅ Required Practices

1. **Guard Clauses**: Use guard clauses instead of deep nesting
   ```python
   # ✅ GOOD
   if not clip_id:
       raise ValueError("clip_id required")
   if len(cameras) < 2:
       raise ValueError("Need at least 2 cameras")
   return process_clip(clip_id, cameras)
   
   # ❌ BAD
   if clip_id:
       if len(cameras) >= 2:
           return process_clip(clip_id, cameras)
       else:
           raise ValueError("Need at least 2 cameras")
   ```

2. **Avoid `else` blocks**: Use early returns
   ```python
   # ✅ GOOD
   if condition:
       return value_a
   return value_b
   
   # ❌ BAD
   if condition:
       return value_a
   else:
       return value_b
   ```

3. **Maximum 2 levels of nesting**: Refactor if deeper
4. **Explicit types everywhere**: No `any` in TypeScript, full type hints in Python
5. **Compact, readable functions**: Single responsibility, <50 lines preferred

### File Structure

- **Backend**: `/backend/` - FastAPI app, API endpoints, orchestration
- **Frontend**: `/frontend/src/` - React components, pages, services
- **ML/CV**: `/ml/` - Detection, tracking, 3D reconstruction, decision engine
- **Tests**: Co-located with code or in `/tests/` directories
- **Documentation**: Markdown files in root

---

## 🎯 PHASE 6: Forward Pass Decision Engine

### Goal
Implement forward-pass decision engine using 3D data and physics-based analysis.

### Tasks Breakdown (Parallelizable)

#### Task 6.1: Decision Criteria Definition
**Agent Assignment**: Physics/ML Agent

**Deliverables**:
- `FORWARD_PASS_PHYSICS_MODEL.md` - Mathematical model documentation
- Define criteria:
  - Simple displacement-based model (Phase 1)
  - Ball displacement along field axis
  - Player momentum consideration (simplified)
  - Extension points for future metric tensor model

**Specifications**:
- Rugby law: Pass is forward if ball travels forward relative to the passer's momentum
- Phase 1: Simple model - compare ball's net forward displacement
- Document physics equations clearly
- Include examples: clearly forward, clearly backward, borderline cases

**Files to Create/Modify**:
- `FORWARD_PASS_PHYSICS_MODEL.md` (new)
- Reference in `ml/decision_engine.py`

---

#### Task 6.2: Pass Event Detection
**Agent Assignment**: ML/CV Agent

**Deliverables**:
- Pass event detection logic in `ml/decision_engine.py`
- Identify pass start time (ball leaves passer's hands)
- Identify pass end time (ball caught/grounded)

**Specifications**:
- Use velocity and position changes as heuristics
- Ball velocity threshold: >5 m/s indicates pass
- Position change: significant movement in short time
- Handle edge cases: ball not caught, multiple touches

**Implementation Notes**:
```python
def detect_pass_events(ball_trajectory_3d: List[Point3D]) -> Tuple[int, int]:
    """
    Detect pass start and end frame indices.
    
    Returns:
        (start_frame, end_frame) - frame indices
    """
    # Use velocity changes to detect pass start
    # Use position stabilization to detect pass end
    pass
```

**Files to Create/Modify**:
- `ml/decision_engine.py` (new)
- `ml/tests/test_decision_engine.py` (new)

---

#### Task 6.3: Ball Trajectory Analysis
**Agent Assignment**: ML/CV Agent

**Deliverables**:
- Trajectory smoothing and analysis functions
- Velocity vector computation from 3D positions
- Handle noisy measurements

**Specifications**:
- Smooth trajectory using Kalman filter OR polynomial fit (start with polynomial)
- Compute velocity: `v = Δposition / Δtime`
- Handle missing 3D positions (interpolation)
- Confidence based on trajectory smoothness

**Implementation Notes**:
```python
def compute_ball_velocity(trajectory: List[Point3D], timestamps: List[float]) -> List[Vector3D]:
    """Compute velocity vectors from 3D trajectory."""
    pass

def smooth_trajectory(trajectory: List[Point3D], window_size: int = 5) -> List[Point3D]:
    """Smooth trajectory using moving average or polynomial fit."""
    pass
```

**Files to Create/Modify**:
- `ml/decision_engine.py` (add functions)
- `ml/tests/test_decision_engine.py` (add tests)

---

#### Task 6.4: Decision Logic Implementation
**Agent Assignment**: Core ML Agent

**Deliverables**:
- Main decision engine class/function
- `DecisionResult` dataclass
- Integration with 3D reconstruction output

**Specifications**:
- Input: 3D ball trajectory, player positions, timestamps
- Output: `DecisionResult(is_forward: bool, confidence: float, explanation: str)`
- Decision criteria:
  1. Compute ball's net forward displacement
  2. Account for passer's momentum (simplified: use passer's velocity)
  3. Compare: ball forward displacement vs passer forward movement
  4. If ball moves forward MORE than passer: FORWARD pass

**DecisionResult Model**:
```python
@dataclass
class DecisionResult:
    is_forward: bool
    confidence: float  # 0.0 to 1.0
    explanation: str  # Human-readable reasoning
    metadata: Optional[Dict[str, Any]] = None  # Debug data
```

**Main Function Signature**:
```python
def analyze_forward_pass(
    ball_trajectory_3d: List[Point3D],
    passer_trajectory_3d: Optional[List[Point3D]],
    timestamps: List[float],
    field_axis_forward: Vector3D
) -> DecisionResult:
    """
    Analyze if pass is forward.
    
    Args:
        ball_trajectory_3d: 3D positions of ball over time
        passer_trajectory_3d: 3D positions of passer (if available)
        timestamps: Time for each position
        field_axis_forward: Forward direction vector on field
    
    Returns:
        DecisionResult with decision and confidence
    """
    pass
```

**Files to Create/Modify**:
- `ml/decision_engine.py` (main implementation)
- `ml/tests/test_decision_engine.py` (comprehensive tests)

---

#### Task 6.5: Confidence Scoring
**Agent Assignment**: ML Agent

**Deliverables**:
- Confidence calculation algorithm
- Factors: detection confidence, 3D quality, trajectory smoothness

**Specifications**:
- Base confidence: detection confidence (from Phase 4)
- 3D quality factor: reprojection error, number of cameras used
- Trajectory smoothness: low variance = higher confidence
- Combine factors: `confidence = base * 3d_quality * smoothness_factor`
- Range: 0.0 to 1.0

**Implementation**:
```python
def compute_confidence(
    detection_confidences: List[float],
    reprojection_errors: List[float],
    trajectory_variance: float,
    num_cameras: int
) -> float:
    """
    Compute overall confidence score.
    
    Returns:
        Confidence between 0.0 and 1.0
    """
    pass
```

**Files to Create/Modify**:
- `ml/decision_engine.py` (add confidence function)
- `ml/tests/test_decision_engine.py` (add confidence tests)

---

#### Task 6.6: Testing and Validation
**Agent Assignment**: Testing Agent

**Deliverables**:
- Comprehensive unit tests
- Synthetic test scenarios
- Integration tests with 3D reconstruction

**Test Scenarios**:
1. **Clearly Forward**: Ball moves 5m forward, passer moves 1m forward → FORWARD
2. **Clearly Backward**: Ball moves 2m backward, passer moves 1m forward → NOT FORWARD
3. **Borderline Forward**: Ball moves 1.5m forward, passer moves 1m forward → FORWARD (low confidence)
4. **Borderline Backward**: Ball moves 0.8m forward, passer moves 1m forward → NOT FORWARD
5. **No Passer Data**: Use ball displacement only → decision with lower confidence
6. **Noisy Trajectory**: Add noise, verify smoothing works
7. **Missing Frames**: Interpolate missing positions

**Files to Create/Modify**:
- `ml/tests/test_decision_engine.py` (comprehensive test suite)
- Test fixtures in `ml/tests/fixtures/` (synthetic trajectories)

---

## 🎯 PHASE 7: Backend API and Pipeline Glue

### Goal
Wire all components into coherent backend API with full orchestration.

### Tasks Breakdown

#### Task 7.1: Main API Endpoint Implementation
**Agent Assignment**: Backend Agent

**Deliverables**:
- Complete `POST /api/clip/analyse-pass` endpoint
- Full pipeline orchestration: ingestion → detection → tracking → 3D → decision

**Current State**: Endpoint exists but is stubbed/mock

**Required Implementation**:
```python
@app.post("/api/clip/analyse-pass", response_model=DecisionResult)
async def analyse_pass(request: AnalysePassRequest):
    """
    Full pipeline:
    1. Ingest and sync video frames
    2. Detect players and ball
    3. Track objects across frames
    4. Reconstruct 3D positions
    5. Analyze forward pass decision
    6. Return result with confidence
    """
    # TODO: Implement full pipeline
    pass
```

**Error Handling**:
- Insufficient cameras (<2): HTTP 400 with clear message
- Ball not detected: Return low confidence decision
- Calibration missing: Use defaults or return error
- Processing timeout: HTTP 504

**Files to Modify**:
- `backend/main.py` (implement endpoint)
- `backend/api/` (create if needed, organize endpoints)

---

#### Task 7.2: Debug Data Endpoint
**Agent Assignment**: Backend Agent

**Deliverables**:
- `GET /api/clip/{clip_id}/debug-data` endpoint
- Return detailed pipeline data for analysts

**Response Format**:
```json
{
  "clip_id": "...",
  "detections": [...],
  "tracks": [...],
  "3d_positions": [...],
  "trajectory": [...],
  "decision_metadata": {...}
}
```

**Files to Modify**:
- `backend/main.py` (add endpoint)

---

#### Task 7.3: Request/Response Models
**Agent Assignment**: Backend Agent

**Deliverables**:
- Refine Pydantic models
- Add validation
- Update OpenAPI docs

**Models to Update**:
- `AnalysePassRequest` - add more validation
- `DecisionResult` - ensure all fields
- New: `DebugDataResponse`

**Files to Modify**:
- `backend/main.py` (update models)

---

#### Task 7.4: Logging and Metrics
**Agent Assignment**: Backend Agent

**Deliverables**:
- Structured logging per pipeline stage
- Latency tracking per component
- Failure/warning counts

**Implementation**:
```python
import logging
import time

logger = logging.getLogger(__name__)

# Per-stage timing
stage_times = {}
start = time.time()
# ... do work ...
stage_times["detection"] = time.time() - start
```

**Files to Modify**:
- `backend/main.py` (add logging)

---

#### Task 7.5: Integration Testing
**Agent Assignment**: Testing Agent

**Deliverables**:
- End-to-end tests with synthetic data
- Mock components for unit tests
- Validate full pipeline flow

**Test Cases**:
1. Happy path: Full pipeline with synthetic data
2. Missing ball detection: Graceful degradation
3. Insufficient cameras: Error handling
4. Timeout: Proper error response

**Files to Create**:
- `backend/tests/test_integration.py` (new)
- `backend/tests/fixtures/` (test data)

---

## 🎯 PHASE 8: Frontend UI for Referees and Analysts

### Goal
Build clean, intuitive UI for TMOs and analysts.

### Tasks Breakdown

#### Task 8.1: Core UI Components
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Video player component (multi-view or switchable)
- Decision indicator (prominent, color-coded: red=FORWARD, green=NOT FORWARD)
- Confidence display (percentage + visual bar)
- Explanation text panel

**Component Structure**:
```
/frontend/src/
  components/
    VideoPlayer.tsx
    DecisionIndicator.tsx
    ConfidenceDisplay.tsx
    ExplanationPanel.tsx
```

**Design Requirements**:
- High contrast for visibility
- Large buttons for TMO use
- Clear, uncluttered interface
- Accessible (WCAG AA)

**Files to Create**:
- `frontend/src/components/VideoPlayer.tsx`
- `frontend/src/components/DecisionIndicator.tsx`
- `frontend/src/components/ConfidenceDisplay.tsx`
- `frontend/src/components/ExplanationPanel.tsx`

---

#### Task 8.2: Timeline Component
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Timeline with pass event markers
- Scrubbing capability
- Frame-by-frame controls

**Features**:
- Show video timeline
- Mark pass start/end times
- Allow scrubbing to any frame
- Play/pause controls
- Step forward/backward buttons

**Files to Create**:
- `frontend/src/components/Timeline.tsx`

---

#### Task 8.3: Analysis Control Panel
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Clip selection/upload
- Camera selection (checkboxes)
- Time window controls (start/end time inputs)
- "Analyse Pass" button

**Component**:
```typescript
interface AnalysisControlProps {
  onAnalyse: (config: AnalysisConfig) => void;
  cameras: string[];
  clipId: string;
}
```

**Files to Create**:
- `frontend/src/components/AnalysisControlPanel.tsx`

---

#### Task 8.4: Results Display
**Agent Assignment**: Frontend Agent

**Deliverables**:
- FORWARD / NOT FORWARD indicator
- Confidence percentage
- Explanation text
- Optional: 2D field view overlay

**Layout**:
- Large, prominent decision indicator
- Confidence bar below
- Explanation text in readable format
- Optional debug toggle for analysts

**Files to Create**:
- `frontend/src/components/ResultsDisplay.tsx`
- `frontend/src/components/FieldView.tsx` (optional)

---

#### Task 8.5: Debug View (Analysts)
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Per-frame detections overlay on video
- 3D trajectory visualization (top-down view)
- Export debug data (JSON download)

**Features**:
- Toggle detections overlay
- Show bounding boxes on video
- Top-down field view with trajectory
- Download debug JSON button

**Files to Create**:
- `frontend/src/components/DebugView.tsx`
- `frontend/src/components/TrajectoryVisualization.tsx`

---

#### Task 8.6: State Management
**Agent Assignment**: Frontend Agent

**Deliverables**:
- React context or hooks for global state
- Handle loading states
- Error state display

**State Structure**:
```typescript
interface AppState {
  clipId: string | null;
  cameras: string[];
  analysisResult: DecisionResult | null;
  isLoading: boolean;
  error: string | null;
}
```

**Files to Create/Modify**:
- `frontend/src/context/AppContext.tsx` (or use hooks)
- `frontend/src/hooks/useAnalysis.ts`

---

#### Task 8.7: Main Page Integration
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Main analysis page combining all components
- Routing setup
- API service integration

**Page Layout**:
```
┌─────────────────────────────────────┐
│  Analysis Control Panel            │
├─────────────────────────────────────┤
│  Video Player    │  Results        │
│                  │  Decision        │
│                  │  Confidence      │
├─────────────────────────────────────┤
│  Timeline                           │
└─────────────────────────────────────┘
```

**Files to Create/Modify**:
- `frontend/src/pages/AnalysisPage.tsx`
- `frontend/src/services/api.ts` (API client)
- `frontend/src/App.tsx` (routing)

---

#### Task 8.8: Testing
**Agent Assignment**: Testing Agent

**Deliverables**:
- Component tests (Jest + React Testing Library)
- E2E tests (Playwright/Cypress)
- Usability considerations

**Test Coverage**:
- All components render correctly
- User interactions work
- API integration works
- Error states display properly

**Files to Create**:
- `frontend/src/components/__tests__/` (component tests)
- `frontend/e2e/` (E2E tests)

---

## 🎯 PHASE 9: Testing Strategy

### Goal
Comprehensive test coverage across all layers.

### Tasks Breakdown

#### Task 9.1: Test Strategy Document
**Agent Assignment**: Testing Agent

**Deliverables**:
- `TEST_STRATEGY.md`
- Coverage targets: 80%+ backend/ML, 70%+ frontend
- Test data management plan

---

#### Task 9.2: Backend Unit Tests
**Agent Assignment**: Testing Agent

**Deliverables**:
- Complete coverage for all backend modules
- Pytest with fixtures
- Target: 80%+ coverage

**Files to Create/Modify**:
- `backend/tests/test_*.py` (comprehensive coverage)

---

#### Task 9.3: ML/CV Unit Tests
**Agent Assignment**: Testing Agent

**Deliverables**:
- Geometry function tests (triangulation, transforms)
- Decision logic tests with known scenarios
- Mock detection/tracking outputs

**Files to Create/Modify**:
- `ml/tests/test_*.py` (all modules)

---

#### Task 9.4: Frontend Unit Tests
**Agent Assignment**: Testing Agent

**Deliverables**:
- Component tests for all UI components
- Service/API client tests
- State management tests

**Files to Create**:
- `frontend/src/**/__tests__/*.test.tsx`

---

#### Task 9.5: E2E Tests
**Agent Assignment**: Testing Agent

**Deliverables**:
- Main workflow: load clip → analyse → view result
- Error scenarios
- Multiple pass analyses

**Files to Create**:
- `frontend/e2e/main-workflow.spec.ts`

---

## 🎯 PHASE 10: CI/CD and Deployment

### Goal
Automated testing and deployment pipeline.

### Tasks Breakdown

#### Task 10.1: CI Configuration
**Agent Assignment**: DevOps Agent

**Deliverables**:
- GitHub Actions workflow
- Steps: lint, type-check, test, build
- Separate jobs for frontend and backend

**Files to Create/Modify**:
- `.github/workflows/ci.yml` (update existing)

---

#### Task 10.2: Docker Configuration
**Agent Assignment**: DevOps Agent

**Deliverables**:
- Optimized Dockerfiles
- Multi-stage builds
- Docker Compose for local dev

**Files to Create/Modify**:
- `backend/Dockerfile` (optimize)
- `frontend/Dockerfile` (optimize)
- `docker-compose.yml` (update)

---

## 🎯 PHASE 11: Datasets and Training (Parallel)

### Goal
Plan data collection and model improvement.

### Tasks Breakdown

#### Task 11.1: Dataset Schema
**Agent Assignment**: ML Agent

**Deliverables**:
- Dataset structure definition
- Annotation format
- Storage plan

---

#### Task 11.2: Labeling Tool
**Agent Assignment**: Frontend Agent

**Deliverables**:
- Simple web UI for labeling passes
- Mark start/end times
- Label FORWARD/NOT FORWARD

---

## 🎯 PHASE 12: Performance Optimization

### Goal
Reduce latency and optimize for near-real-time.

### Tasks Breakdown

#### Task 12.1: Performance Profiling
**Agent Assignment**: Performance Agent

**Deliverables**:
- Identify bottlenecks
- Measure per-stage latency
- Profiling reports

---

#### Task 12.2: Optimization
**Agent Assignment**: Performance Agent

**Deliverables**:
- GPU acceleration
- Model quantization
- Frame subsampling
- Parallel processing

---

## 🎯 PHASE 13: Documentation and Demo

### Goal
Package for stakeholders and users.

### Tasks Breakdown

#### Task 13.1: User Documentation
**Agent Assignment**: Documentation Agent

**Deliverables**:
- User guide
- API documentation
- Deployment guide

---

#### Task 13.2: Demo Materials
**Agent Assignment**: Documentation Agent

**Deliverables**:
- Demo video
- One-pager
- Pitch deck

---

## 📝 Implementation Guidelines

### Code Quality Checklist

Before submitting work, verify:
- [ ] Guard clauses used (no deep nesting)
- [ ] No unnecessary `else` blocks
- [ ] Maximum 2 levels of nesting
- [ ] Full type hints/TypeScript types
- [ ] Functions <50 lines (preferred)
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Follows existing code style

### Testing Requirements

- **Unit Tests**: Required for all new functions/classes
- **Integration Tests**: Required for API endpoints
- **E2E Tests**: Required for main user workflows
- **Coverage**: Aim for 80%+ backend/ML, 70%+ frontend

### Documentation Requirements

- **Code Comments**: Docstrings for all public functions/classes
- **README Updates**: Update relevant README files
- **API Docs**: OpenAPI/Swagger auto-generated from FastAPI
- **Architecture Docs**: Update if architecture changes

### Git Workflow

1. Create feature branch: `feature/phase6-decision-engine`
2. Make changes following coding standards
3. Write tests
4. Commit with clear messages
5. Push and create PR
6. Ensure CI passes

---

## 🚀 Quick Start for Agents

### For Backend/ML Agents:

1. **Review existing code**:
   - `backend/main.py` - Current API structure
   - `ml/detector.py`, `ml/tracker.py` - Detection/tracking
   - `ml/triangulation.py`, `ml/spatial_model.py` - 3D reconstruction

2. **Understand data flow**:
   - Video → Detection → Tracking → 3D → Decision

3. **Start with Phase 6**:
   - Create `ml/decision_engine.py`
   - Implement decision logic
   - Write tests
   - Integrate with backend

### For Frontend Agents:

1. **Review existing structure**:
   - `frontend/src/App.tsx` - Current app structure
   - `frontend/src/services/` - API client (if exists)

2. **Understand API**:
   - `POST /api/clip/analyse-pass` - Main endpoint
   - `GET /api/clip/{id}/debug-data` - Debug endpoint

3. **Start with Phase 8**:
   - Create components in `frontend/src/components/`
   - Build main analysis page
   - Integrate with API

### For Testing Agents:

1. **Review test structure**:
   - `backend/tests/` - Backend tests
   - `ml/tests/` - ML tests
   - `frontend/src/**/__tests__/` - Frontend tests

2. **Start with Phase 9**:
   - Write comprehensive test suites
   - Set up CI integration
   - Achieve coverage targets

---

## 📚 Key Documents Reference

- **PLAN_RUGBY_VISION.md** - Complete phase breakdown
- **ARCHITECTURE_OVERVIEW.md** - System architecture
- **RUGBY_VISION_REQUIREMENTS.md** - Functional requirements
- **CONTRIBUTING.md** - Coding standards (CRITICAL)
- **PHASE_4_IMPLEMENTATION_SUMMARY.md** - What's been done

---

## ✅ Success Criteria Summary

### Phase 6 (Decision Engine)
- [ ] 100% correct on obvious test cases (10/10)
- [ ] 70%+ correct on borderline cases
- [ ] Confidence scores align with accuracy
- [ ] <1 second computation time per pass
- [ ] All tests pass

### Phase 7 (Backend API)
- [ ] API endpoints functional and documented
- [ ] End-to-end test passes with synthetic data
- [ ] Latency < 10 seconds for 5-second clip
- [ ] Error handling tested and robust

### Phase 8 (Frontend UI)
- [ ] TMO can use interface with <5 minutes training
- [ ] All core workflows functional
- [ ] UI tests pass
- [ ] Responsive on common screen sizes

### Phase 9 (Testing)
- [ ] 80%+ code coverage (backend and ML)
- [ ] 70%+ code coverage (frontend)
- [ ] All tests pass in CI
- [ ] Test execution < 5 minutes

---

## 🎯 Next Immediate Steps

1. **Phase 6 - Decision Engine** (HIGHEST PRIORITY)
   - Start with Task 6.1: Decision Criteria Definition
   - Then Task 6.4: Decision Logic Implementation
   - Parallel: Tasks 6.2, 6.3, 6.5, 6.6

2. **Phase 7 - Backend Integration** (After Phase 6)
   - Wire Phase 6 into backend API
   - Full pipeline orchestration

3. **Phase 8 - Frontend UI** (Can start in parallel with Phase 7)
   - Build components
   - Integrate with API

---

## 📞 Questions or Issues?

- Review existing code and documentation first
- Check `CONTRIBUTING.md` for coding standards
- Follow the architecture patterns established
- Write tests for all new code
- Update documentation as you go

---

**Let's build Rugby Vision! 🏉**

