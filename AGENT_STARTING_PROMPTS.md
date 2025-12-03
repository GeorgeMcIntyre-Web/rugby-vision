# Agent Starting Prompts

Copy and paste these prompts to assign agents to specific tasks. Each prompt directs the agent to the `MEGA_PROMPT_FOR_AGENTS.md` file for detailed specifications.

---

## Phase 6: Forward Pass Decision Engine

### Physics/ML Agent - Task 6.1: Decision Criteria Definition

```
You are a Physics/ML Agent working on Rugby Vision Phase 6, Task 6.1: Decision Criteria Definition.

Your task: Define the mathematical model for forward pass detection in rugby.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.1: Decision Criteria Definition"
3. Follow the specifications to create FORWARD_PASS_PHYSICS_MODEL.md
4. Define the mathematical criteria for forward pass detection
5. Document the physics equations clearly
6. Include examples: clearly forward, clearly backward, borderline cases

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### ML/CV Agent - Task 6.2: Pass Event Detection

```
You are a ML/CV Agent working on Rugby Vision Phase 6, Task 6.2: Pass Event Detection.

Your task: Implement pass event detection logic to identify when a pass starts and ends.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.2: Pass Event Detection"
3. Follow the specifications to implement pass event detection in ml/decision_engine.py
4. Identify pass start time (ball leaves passer's hands)
5. Identify pass end time (ball caught/grounded)
6. Use velocity and position changes as heuristics
7. Write unit tests in ml/tests/test_decision_engine.py

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### ML/CV Agent - Task 6.3: Ball Trajectory Analysis

```
You are a ML/CV Agent working on Rugby Vision Phase 6, Task 6.3: Ball Trajectory Analysis.

Your task: Implement ball trajectory smoothing and velocity computation from 3D positions.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.3: Ball Trajectory Analysis"
3. Follow the specifications to implement trajectory analysis in ml/decision_engine.py
4. Smooth trajectory using polynomial fit or Kalman filter
5. Compute velocity vectors from 3D positions
6. Handle noisy measurements and missing positions
7. Write unit tests in ml/tests/test_decision_engine.py

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Core ML Agent - Task 6.4: Decision Logic Implementation

```
You are a Core ML Agent working on Rugby Vision Phase 6, Task 6.4: Decision Logic Implementation.

Your task: Implement the main decision engine that determines if a pass is forward.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.4: Decision Logic Implementation"
3. Follow the specifications to create ml/decision_engine.py with the main decision logic
4. Implement analyze_forward_pass() function
5. Create DecisionResult dataclass
6. Compute ball's net forward displacement
7. Account for passer's momentum
8. Return decision with confidence and explanation
9. Write comprehensive tests in ml/tests/test_decision_engine.py

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### ML Agent - Task 6.5: Confidence Scoring

```
You are a ML Agent working on Rugby Vision Phase 6, Task 6.5: Confidence Scoring.

Your task: Implement confidence calculation algorithm for forward pass decisions.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.5: Confidence Scoring"
3. Follow the specifications to implement confidence scoring in ml/decision_engine.py
4. Factor in: detection confidence, 3D reconstruction quality, trajectory smoothness
5. Combine factors to produce 0.0-1.0 confidence score
6. Write unit tests in ml/tests/test_decision_engine.py

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Testing Agent - Task 6.6: Testing and Validation

```
You are a Testing Agent working on Rugby Vision Phase 6, Task 6.6: Testing and Validation.

Your task: Create comprehensive test suite for the decision engine.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 6: Forward Pass Decision Engine" → "Task 6.6: Testing and Validation"
3. Follow the specifications to create comprehensive tests in ml/tests/test_decision_engine.py
4. Create test scenarios: clearly forward, clearly backward, borderline cases
5. Test with synthetic trajectories
6. Validate confidence scores align with accuracy
7. Create test fixtures in ml/tests/fixtures/

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-5 complete. You're working on Phase 6.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 7: Backend API and Pipeline Glue

### Backend Agent - Task 7.1: Main API Endpoint Implementation

```
You are a Backend Agent working on Rugby Vision Phase 7, Task 7.1: Main API Endpoint Implementation.

Your task: Implement the complete POST /api/clip/analyse-pass endpoint with full pipeline orchestration.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 7: Backend API and Pipeline Glue" → "Task 7.1: Main API Endpoint Implementation"
3. Follow the specifications to implement the full pipeline in backend/main.py
4. Wire together: ingestion → detection → tracking → 3D → decision
5. Add proper error handling (guard clauses)
6. Return DecisionResult with proper HTTP status codes
7. Add structured logging per pipeline stage
8. Write integration tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-6 complete. You're working on Phase 7.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Backend Agent - Task 7.2: Debug Data Endpoint

```
You are a Backend Agent working on Rugby Vision Phase 7, Task 7.2: Debug Data Endpoint.

Your task: Implement GET /api/clip/{clip_id}/debug-data endpoint for analysts.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 7: Backend API and Pipeline Glue" → "Task 7.2: Debug Data Endpoint"
3. Follow the specifications to implement the debug endpoint in backend/main.py
4. Return detailed pipeline data: detections, tracks, 3D positions, trajectory, decision metadata
5. Create DebugDataResponse model
6. Write tests for the endpoint

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-6 complete. You're working on Phase 7.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Backend Agent - Task 7.3: Request/Response Models

```
You are a Backend Agent working on Rugby Vision Phase 7, Task 7.3: Request/Response Models.

Your task: Refine Pydantic models with proper validation and update OpenAPI docs.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 7: Backend API and Pipeline Glue" → "Task 7.3: Request/Response Models"
3. Follow the specifications to update models in backend/main.py
4. Add validation to AnalysePassRequest
5. Ensure DecisionResult has all required fields
6. Create DebugDataResponse model
7. Verify OpenAPI/Swagger docs are accurate

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-6 complete. You're working on Phase 7.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Backend Agent - Task 7.4: Logging and Metrics

```
You are a Backend Agent working on Rugby Vision Phase 7, Task 7.4: Logging and Metrics.

Your task: Add structured logging and latency tracking per pipeline stage.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 7: Backend API and Pipeline Glue" → "Task 7.4: Logging and Metrics"
3. Follow the specifications to add logging in backend/main.py
4. Add structured logging per pipeline stage
5. Track latency per component (detection, tracking, 3D, decision)
6. Count failures and warnings
7. Use Python logging module with appropriate levels

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-6 complete. You're working on Phase 7.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

### Testing Agent - Task 7.5: Integration Testing

```
You are a Testing Agent working on Rugby Vision Phase 7, Task 7.5: Integration Testing.

Your task: Create end-to-end integration tests for the full pipeline.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 7: Backend API and Pipeline Glue" → "Task 7.5: Integration Testing"
3. Follow the specifications to create integration tests in backend/tests/test_integration.py
4. Test full pipeline with synthetic data
5. Test error scenarios (missing ball, insufficient cameras, timeouts)
6. Mock components where appropriate for unit tests
7. Create test fixtures in backend/tests/fixtures/

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-6 complete. You're working on Phase 7.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 8: Frontend UI for Referees and Analysts

### Frontend Agent - Task 8.1: Core UI Components

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.1: Core UI Components.

Your task: Create core UI components: VideoPlayer, DecisionIndicator, ConfidenceDisplay, ExplanationPanel.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.1: Core UI Components"
3. Follow the specifications to create components in frontend/src/components/
4. Create VideoPlayer.tsx (multi-view or switchable)
5. Create DecisionIndicator.tsx (red=FORWARD, green=NOT FORWARD)
6. Create ConfidenceDisplay.tsx (percentage + visual bar)
7. Create ExplanationPanel.tsx (human-readable text)
8. Follow design requirements: high contrast, large buttons, accessible
9. Write component tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.2: Timeline Component

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.2: Timeline Component.

Your task: Create timeline component with scrubbing and frame-by-frame controls.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.2: Timeline Component"
3. Follow the specifications to create Timeline.tsx in frontend/src/components/
4. Show video timeline with pass event markers
5. Implement scrubbing capability
6. Add frame-by-frame controls (play/pause, step forward/backward)
7. Write component tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.3: Analysis Control Panel

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.3: Analysis Control Panel.

Your task: Create analysis control panel with clip selection, camera selection, and time controls.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.3: Analysis Control Panel"
3. Follow the specifications to create AnalysisControlPanel.tsx in frontend/src/components/
4. Implement clip selection/upload
5. Add camera selection (checkboxes)
6. Add time window controls (start/end time inputs)
7. Add "Analyse Pass" button
8. Write component tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.4: Results Display

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.4: Results Display.

Your task: Create results display component showing decision, confidence, and explanation.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.4: Results Display"
3. Follow the specifications to create ResultsDisplay.tsx in frontend/src/components/
4. Display FORWARD / NOT FORWARD indicator (large, prominent)
5. Show confidence percentage and visual bar
6. Display explanation text in readable format
7. Optional: Create FieldView.tsx for 2D field overlay
8. Write component tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.5: Debug View (Analysts)

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.5: Debug View.

Your task: Create debug view component for analysts with detections overlay and trajectory visualization.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.5: Debug View"
3. Follow the specifications to create DebugView.tsx and TrajectoryVisualization.tsx
4. Implement per-frame detections overlay on video
5. Create top-down field view with 3D trajectory
6. Add export debug data button (JSON download)
7. Write component tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.6: State Management

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.6: State Management.

Your task: Implement state management using React context or hooks.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.6: State Management"
3. Follow the specifications to create state management
4. Create AppContext.tsx or use hooks for global state
5. Handle loading states
6. Handle error states with clear, non-technical messages
7. Write tests for state management

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Frontend Agent - Task 8.7: Main Page Integration

```
You are a Frontend Agent working on Rugby Vision Phase 8, Task 8.7: Main Page Integration.

Your task: Create main analysis page combining all components and integrate with API.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.7: Main Page Integration"
3. Follow the specifications to create AnalysisPage.tsx
4. Combine all components into main page layout
5. Create API service client in frontend/src/services/api.ts
6. Integrate with POST /api/clip/analyse-pass endpoint
7. Set up routing in App.tsx
8. Write E2E tests for main workflow

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

### Testing Agent - Task 8.8: Frontend Testing

```
You are a Testing Agent working on Rugby Vision Phase 8, Task 8.8: Frontend Testing.

Your task: Create comprehensive test suite for frontend components and E2E tests.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 8: Frontend UI" → "Task 8.8: Testing"
3. Follow the specifications to create tests
4. Write component tests using Jest + React Testing Library
5. Write E2E tests using Playwright or Cypress
6. Test all user interactions
7. Test API integration
8. Test error states
9. Achieve 70%+ code coverage

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-7 complete. You're working on Phase 8.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full TypeScript types).
```

---

## Phase 9: Testing Strategy

### Testing Agent - Phase 9

```
You are a Testing Agent working on Rugby Vision Phase 9: Testing Strategy.

Your task: Create comprehensive test strategy and achieve coverage targets.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 9: Testing Strategy"
3. Follow the specifications to create comprehensive test suites
4. Write TEST_STRATEGY.md document
5. Complete backend unit tests (target: 80%+ coverage)
6. Complete ML/CV unit tests (target: 80%+ coverage)
7. Complete frontend unit tests (target: 70%+ coverage)
8. Create E2E test suite
9. Set up CI integration for tests

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-8 complete. You're working on Phase 9.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 10: CI/CD and Deployment

### DevOps Agent - Phase 10

```
You are a DevOps Agent working on Rugby Vision Phase 10: CI/CD and Deployment Pipeline.

Your task: Set up continuous integration and deployment infrastructure.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 10: CI/CD and Deployment Pipeline"
3. Follow the specifications to set up CI/CD
4. Configure GitHub Actions workflow (lint, type-check, test, build)
5. Optimize Dockerfiles (multi-stage builds)
6. Update docker-compose.yml for local dev
7. Create deployment scripts
8. Set up health checks
9. Write CI_CD_SETUP.md documentation

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-9 complete. You're working on Phase 10.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 11: Datasets and Training

### ML Agent - Phase 11

```
You are a ML Agent working on Rugby Vision Phase 11: Datasets, Labeling, and Model Training.

Your task: Plan and scaffold data collection, labeling, and model improvement.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 11: Datasets, Labeling, and Model Training"
3. Follow the specifications to create data pipeline
4. Design dataset schema
5. Create data ingestion pipeline
6. Create labeling tool (web UI)
7. Set up training scripts in ml/training/
8. Write DATASET_AND_LABELING_PLAN.md

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-10 complete. You're working on Phase 11.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 12: Performance Optimization

### Performance Agent - Phase 12

```
You are a Performance Agent working on Rugby Vision Phase 12: Performance, Latency, and Real-Time.

Your task: Optimize system for near-real-time operation and reduce latency.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 12: Performance Optimization"
3. Follow the specifications to optimize performance
4. Profile system to identify bottlenecks
5. Implement GPU acceleration
6. Add model quantization
7. Implement frame subsampling
8. Add parallel processing
9. Implement RTSP/RTMP stream support
10. Write PERFORMANCE_AND_REALTIME_PLAN.md

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-11 complete. You're working on Phase 12.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Phase 13: Documentation and Demo

### Documentation Agent - Phase 13

```
You are a Documentation Agent working on Rugby Vision Phase 13: Documentation, Demo, and Pitch Material.

Your task: Package Rugby Vision for stakeholders, customers, and users.

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "Phase 13: Documentation and Demo"
3. Follow the specifications to create documentation
4. Write user guide (RUGBY_VISION_USER_GUIDE.md)
5. Write technical overview (RUGBY_VISION_TECH_OVERVIEW.md)
6. Create demo script and video
7. Create one-pager (PDF)
8. Create pitch deck (slides)
9. Update API documentation
10. Write deployment guide

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: Phases 1-12 complete. You're working on Phase 13.

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## General Agent Prompt Template

If you need to assign a custom task, use this template:

```
You are a [AGENT_TYPE] Agent working on Rugby Vision [PHASE/TASK].

Your task: [BRIEF DESCRIPTION]

Instructions:
1. Open and read MEGA_PROMPT_FOR_AGENTS.md
2. Navigate to "[PHASE/TASK SECTION]"
3. Follow the specifications to [WHAT TO DO]
4. [SPECIFIC REQUIREMENTS]
5. Write tests as required
6. Follow Git workflow to commit changes

Repository: https://github.com/GeorgeMcIntyre-Web/rugby-vision
Current Status: [RELEVANT PHASES] complete. You're working on [CURRENT PHASE].

Follow the coding standards in MEGA_PROMPT_FOR_AGENTS.md (guard clauses, no deep nesting, full type hints).
```

---

## Usage Tips

1. **Copy the entire prompt** for the agent you want to assign
2. **Paste it directly** to the agent (e.g., in Abacus.ai, ChatGPT, Claude, etc.)
3. **The agent will**:
   - Read MEGA_PROMPT_FOR_AGENTS.md
   - Navigate to their assigned section
   - Follow the detailed specifications
   - Implement according to coding standards
   - Write tests
   - Commit changes

4. **You can assign multiple agents** in parallel for different tasks
5. **Each agent works independently** but follows the same standards

---

**Note**: All prompts reference the same MEGA_PROMPT_FOR_AGENTS.md file, ensuring consistency across all agents.

