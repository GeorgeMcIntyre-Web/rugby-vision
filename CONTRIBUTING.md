# Contributing to Rugby Vision

## Coding Standards

This project follows strict coding standards to ensure maintainability, readability, and consistency across all code. **All contributors must follow these rules.**

## Core Coding Rules

### 1. Use Guard Clauses (No Deep Nesting)

**❌ Bad:**
```python
def process_video(video_path):
    if video_path:
        if os.path.exists(video_path):
            if video_path.endswith('.mp4'):
                # Process video
                return process(video_path)
            else:
                return None
        else:
            return None
    else:
        return None
```

**✅ Good:**
```python
def process_video(video_path: str) -> Optional[VideoData]:
    if not video_path:
        return None
    
    if not os.path.exists(video_path):
        return None
    
    if not video_path.endswith('.mp4'):
        return None
    
    return process(video_path)
```

### 2. Avoid `else` / `elif` When Possible

**❌ Bad:**
```typescript
function getStatus(code: number): string {
    if (code === 200) {
        return 'success';
    } else if (code === 404) {
        return 'not found';
    } else if (code === 500) {
        return 'error';
    } else {
        return 'unknown';
    }
}
```

**✅ Good:**
```typescript
function getStatus(code: number): string {
    if (code === 200) return 'success';
    if (code === 404) return 'not found';
    if (code === 500) return 'error';
    return 'unknown';
}
```

Or use a lookup structure:

```typescript
function getStatus(code: number): string {
    const statusMap: Record<number, string> = {
        200: 'success',
        404: 'not found',
        500: 'error',
    };
    return statusMap[code] ?? 'unknown';
}
```

### 3. Maximum 2 Levels of Nesting

**❌ Bad (3+ levels):**
```python
def analyse_frames(frames):
    results = []
    for frame in frames:
        if frame.is_valid():
            for detection in frame.detections:
                if detection.confidence > 0.5:
                    results.append(detection)  # Level 3
    return results
```

**✅ Good (max 2 levels):**
```python
def analyse_frames(frames: List[Frame]) -> List[Detection]:
    results: List[Detection] = []
    
    for frame in frames:
        if not frame.is_valid():
            continue
        
        valid_detections = get_valid_detections(frame)
        results.extend(valid_detections)
    
    return results

def get_valid_detections(frame: Frame) -> List[Detection]:
    return [
        detection 
        for detection in frame.detections 
        if detection.confidence > 0.5
    ]
```

### 4. Explicit Types (No `any` or Implicit Types)

**❌ Bad:**
```typescript
// TypeScript
function processData(data: any) {
    return data.results;
}

// Python
def process_data(data):
    return data['results']
```

**✅ Good:**
```typescript
// TypeScript
interface ApiResponse {
    results: string[];
    status: number;
}

function processData(data: ApiResponse): string[] {
    return data.results;
}

// Python
from typing import Dict, List

def process_data(data: Dict[str, List[str]]) -> List[str]:
    return data['results']
```

### 5. Clear, Compact, Readable Functions

**Guidelines:**
- One function = one responsibility
- Maximum ~50 lines per function (guideline, not strict rule)
- Descriptive names (verb + noun)
- Document complex logic with docstrings

**✅ Good:**
```python
def synchronize_camera_frames(
    frames: List[CameraFrame],
    reference_timestamp: float
) -> List[CameraFrame]:
    """Synchronize frames from multiple cameras to a reference timestamp.
    
    Args:
        frames: List of frames from different cameras
        reference_timestamp: Target timestamp for synchronization
        
    Returns:
        List of synchronized frames, one per camera
        
    Raises:
        ValueError: If frames list is empty
    """
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    return [
        find_closest_frame(camera_frames, reference_timestamp)
        for camera_frames in group_by_camera(frames)
    ]
```

## Language-Specific Standards

### Python

#### Style
- **Formatter**: Black (line length: 88)
- **Linter**: Flake8
- **Type Checker**: mypy (strict mode)

#### Type Hints
- All function signatures must have complete type hints
- Use `typing` module for complex types
- Enable `--strict` mode in mypy

```python
from typing import List, Optional, Dict, Tuple

def example(
    param1: str,
    param2: List[int],
    param3: Optional[Dict[str, float]] = None
) -> Tuple[bool, str]:
    ...
```

#### Docstrings
- Use Google-style docstrings
- Document all public functions and classes
- Include Args, Returns, Raises sections

```python
def load_video(path: str, start_frame: int = 0) -> VideoCapture:
    """Load a video file from the given path.
    
    Args:
        path: Absolute path to the video file
        start_frame: Frame number to start reading from
        
    Returns:
        OpenCV VideoCapture object ready for frame reading
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If start_frame is negative
    """
    ...
```

### TypeScript/React

#### Style
- **Formatter**: ESLint + Prettier
- **Config**: Strict TypeScript config (see `tsconfig.json`)

#### Type Safety
- Never use `any` type
- Use `unknown` if type is truly unknown, then narrow it
- Explicit return types for all functions
- Interface/Type for all object shapes

```typescript
interface VideoClip {
    id: string;
    cameras: string[];
    startTime: number;
    endTime: number;
}

function analyseClip(clip: VideoClip): Promise<DecisionResult> {
    // Guard clause
    if (clip.endTime <= clip.startTime) {
        throw new Error('Invalid time range');
    }
    
    // Implementation...
}
```

#### React Components
- Functional components only
- Explicit prop types
- Use hooks correctly

```typescript
interface VideoPlayerProps {
    videoUrl: string;
    onTimeUpdate: (time: number) => void;
}

function VideoPlayer({ videoUrl, onTimeUpdate }: VideoPlayerProps): React.ReactElement {
    // Guard clause
    if (!videoUrl) {
        return <div>No video URL provided</div>;
    }
    
    // Component logic...
    return <video src={videoUrl} />;
}
```

## Testing Standards

### Test Structure
- Arrange-Act-Assert pattern
- One assertion per test (guideline)
- Descriptive test names

```python
def test_video_sync_aligns_frames_correctly():
    # Arrange
    frames = create_test_frames()
    reference_time = 10.5
    
    # Act
    synced = synchronize_camera_frames(frames, reference_time)
    
    # Assert
    assert len(synced) == 3
    assert all(abs(f.timestamp - reference_time) < 0.1 for f in synced)
```

### Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for decision logic
- Mock external dependencies

## Git Workflow

### Branch Naming
- `feature/short-description`
- `bugfix/issue-description`
- `docs/what-changed`

### Commit Messages
- Use conventional commits format
- Present tense, imperative mood

```
feat: add video synchronization logic
fix: correct timestamp alignment bug
docs: update architecture overview
test: add unit tests for decision engine
```

### Pull Requests
- Reference related issues
- Include description of changes
- Ensure all tests pass
- Update documentation if needed

## Code Review Checklist

**Before submitting PR:**
- [ ] All tests pass
- [ ] Code follows guard clause pattern
- [ ] No `else` blocks (unless truly necessary)
- [ ] Maximum 2 levels of nesting
- [ ] All types explicitly declared
- [ ] Functions are compact and focused
- [ ] Docstrings/comments for complex logic
- [ ] No linter warnings
- [ ] Type checker passes (mypy/tsc)

**For reviewers:**
- [ ] Code is readable and maintainable
- [ ] Logic is correct and handles edge cases
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] Performance considerations addressed

## Common Patterns

### Error Handling

**Python:**
```python
def safe_operation(data: Dict[str, Any]) -> Optional[Result]:
    # Guard clauses for validation
    if not data:
        logger.error("Data is empty")
        return None
    
    if 'required_field' not in data:
        logger.error("Missing required field")
        return None
    
    try:
        return perform_operation(data)
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        return None
```

**TypeScript:**
```typescript
function safeOperation(data: ApiData): Result | null {
    // Guard clauses
    if (!data) {
        console.error('Data is empty');
        return null;
    }
    
    if (!data.requiredField) {
        console.error('Missing required field');
        return null;
    }
    
    try {
        return performOperation(data);
    } catch (error) {
        console.error('Operation failed:', error);
        return null;
    }
}
```

### List Processing

**Prefer comprehensions/filters over loops:**

```python
# Good
valid_frames = [f for f in frames if f.is_valid()]
high_conf = [d for d in detections if d.confidence > 0.8]

# Also good for complex logic
def get_valid_frames(frames: List[Frame]) -> List[Frame]:
    return [frame for frame in frames if frame.is_valid()]
```

## Questions?

If you're unsure about how to apply these rules to a specific situation:
1. Check existing code for examples
2. Ask in PR comments
3. Refer to this guide

When in doubt, prioritize **readability** and **maintainability**.
