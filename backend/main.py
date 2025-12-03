"""Main FastAPI application for Rugby Vision backend."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import sys
import os

# Add parent directory to path for ml imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rugby Vision API",
    description="Multi-camera 3D forward pass detection system for rugby",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysePassRequest(BaseModel):
    """Request model for pass analysis."""
    clip_id: str = Field(..., description="Unique identifier for the video clip")
    cameras: List[str] = Field(..., description="List of camera IDs to use")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")


class DecisionResult(BaseModel):
    """Result model for pass decision."""
    is_forward: bool = Field(..., description="Whether the pass is forward")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    explanation: str = Field(..., description="Human-readable explanation")
    metadata: Optional[dict] = Field(default=None, description="Additional debug data")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Rugby Vision API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/clip/analyse-pass", response_model=DecisionResult)
async def analyse_pass(request: AnalysePassRequest) -> DecisionResult:
    """Analyse a pass from multi-camera video.
    
    Args:
        request: Analysis request with clip ID, cameras, and time window
        
    Returns:
        DecisionResult with forward/not forward decision and confidence
        
    Raises:
        HTTPException: If validation fails or processing errors occur
    """
    # Guard clause: validate time range
    if request.end_time <= request.start_time:
        raise HTTPException(
            status_code=400,
            detail="end_time must be greater than start_time"
        )
    
    # Guard clause: validate cameras
    if len(request.cameras) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 cameras required for 3D reconstruction"
        )
    
    logger.info(
        f"Analysing pass: clip={request.clip_id}, "
        f"cameras={request.cameras}, "
        f"time=[{request.start_time}, {request.end_time}]"
    )
    
    # Phase 4: Integrate detection and tracking
    # TODO: Phase 5-6: Wire up 3D reconstruction → decision pipeline
    # For now, we return mock decision result
    # But detection/tracking is available via /api/clip/detect-and-track
    
    return DecisionResult(
        is_forward=False,
        confidence=0.85,
        explanation=(
            "Phase 4: Detection and tracking implemented. "
            f"Analysed {len(request.cameras)} camera views over "
            f"{request.end_time - request.start_time:.1f} seconds. "
            "Detection and tracking complete (see /api/clip/detect-and-track endpoint). "
            "3D reconstruction and decision logic coming in Phase 5-6."
        ),
        metadata={
            "clip_id": request.clip_id,
            "cameras_used": request.cameras,
            "duration_seconds": request.end_time - request.start_time,
            "phase": "Phase 4 (Detection & Tracking)",
        }
    )


@app.get("/api/clip/{clip_id}/debug-data")
async def get_debug_data(clip_id: str) -> dict[str, str]:
    """Get debug data for a specific clip.
    
    Args:
        clip_id: Unique identifier for the clip
        
    Returns:
        Debug data dictionary
    """
    logger.info(f"Fetching debug data for clip: {clip_id}")
    
    # Phase 4: Basic implementation
    return {
        "clip_id": clip_id,
        "status": "Use /api/clip/detect-and-track for detection/tracking results",
        "phase": "Phase 4 (Detection & Tracking)",
    }


class DetectAndTrackRequest(BaseModel):
    """Request model for detection and tracking."""
    clip_id: str = Field(..., description="Unique identifier for the video clip")
    cameras: List[str] = Field(..., description="List of camera IDs to use")
    num_frames: int = Field(10, ge=1, le=100, description="Number of frames to process per camera")


@app.post("/api/clip/detect-and-track")
async def detect_and_track(request: DetectAndTrackRequest) -> Dict[str, Any]:
    """Run detection and tracking on a video clip with mock data.
    
    This endpoint demonstrates Phase 4 detection and tracking capabilities.
    Uses mock/synthetic video frames for demonstration.
    
    Args:
        request: Detection and tracking request with clip ID and cameras
        
    Returns:
        Detection and tracking results with summary statistics
    """
    # Guard clause: validate cameras
    if not request.cameras:
        raise HTTPException(
            status_code=400,
            detail="At least one camera required"
        )
    
    logger.info(
        f"Running detection and tracking: clip={request.clip_id}, "
        f"cameras={request.cameras}, frames={request.num_frames}"
    )
    
    try:
        # Import here to avoid errors if ml module not in path
        from ml.detector import Detector
        from ml.tracker import Tracker
        from ml.detection_tracking_api import (
            ClipDefinition,
            run_detection_and_tracking,
            get_detections_summary,
        )
        import numpy as np
        
        # Create mock frames for demonstration
        frames_per_camera: Dict[str, List[np.ndarray]] = {}
        for camera_id in request.cameras:
            frames = [
                np.zeros((720, 1280, 3), dtype=np.uint8)
                for _ in range(request.num_frames)
            ]
            frames_per_camera[camera_id] = frames
        
        # Create clip definition
        clip = ClipDefinition(
            clip_id=request.clip_id,
            camera_ids=request.cameras,
            frames_per_camera=frames_per_camera,
            start_frame=0,
            end_frame=request.num_frames,
        )
        
        # Run detection and tracking
        result = run_detection_and_tracking(clip)
        
        # Get summary
        summary = get_detections_summary(result)
        
        # Format response
        response = {
            "clip_id": result.clip_id,
            "summary": summary,
            "detections_per_camera": {
                camera_id: len(detections)
                for camera_id, detections in result.detections_per_camera.items()
            },
            "tracks_per_camera": {
                camera_id: [
                    {
                        "track_id": track.track_id,
                        "class": track.class_name,
                        "length": track.length,
                        "is_active": track.is_active,
                    }
                    for track in tracks
                ]
                for camera_id, tracks in result.tracks_per_camera.items()
            },
            "metadata": {
                "phase": "Phase 4 (Detection & Tracking)",
                "note": "Using mock detector with synthetic detections",
                "frames_processed": result.frame_count,
            },
        }
        
        return response
        
    except ImportError as e:
        logger.error(f"Failed to import ml modules: {e}")
        raise HTTPException(
            status_code=500,
            detail="ML modules not available. Check system configuration."
        )
    except Exception as e:
        logger.error(f"Detection and tracking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection and tracking failed: {str(e)}"
        )


class Reconstruct3DRequest(BaseModel):
    """Request model for 3D reconstruction."""
    clip_id: str = Field(..., description="Unique identifier for the video clip")
    cameras: List[str] = Field(..., description="List of camera IDs to use")
    num_frames: int = Field(10, ge=1, le=100, description="Number of frames to process")
    calibration_file: Optional[str] = Field(
        default="config/camera_calibration_example.json",
        description="Path to camera calibration file"
    )


@app.post("/api/clip/reconstruct-3d")
async def reconstruct_3d(request: Reconstruct3DRequest) -> Dict[str, Any]:
    """Reconstruct 3D positions from multi-camera tracking with synthetic data.
    
    This endpoint demonstrates Phase 5 3D reconstruction capabilities.
    Pipeline: video ingest → detection/tracking → 3D reconstruction → field coordinates.
    
    Args:
        request: 3D reconstruction request with clip ID, cameras, and calibration
        
    Returns:
        List of FrameState objects with 3D positions in field coordinates
    """
    # Guard clause: validate cameras
    if len(request.cameras) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 cameras required for 3D reconstruction"
        )
    
    logger.info(
        f"Running 3D reconstruction: clip={request.clip_id}, "
        f"cameras={request.cameras}, frames={request.num_frames}"
    )
    
    try:
        # Import Phase 4 modules
        from ml.detector import Detector
        from ml.tracker import Tracker
        from ml.detection_tracking_api import (
            ClipDefinition,
            run_detection_and_tracking,
        )
        
        # Import Phase 5 modules
        from ml.calibration import load_calibration_from_file
        from ml.field_coords import get_standard_rugby_field
        from ml.spatial_model import build_frame_states, get_frame_state_summary
        import numpy as np
        from pathlib import Path
        
        # Create mock frames for demonstration
        frames_per_camera: Dict[str, List[np.ndarray]] = {}
        for camera_id in request.cameras:
            frames = [
                np.zeros((720, 1280, 3), dtype=np.uint8)
                for _ in range(request.num_frames)
            ]
            frames_per_camera[camera_id] = frames
        
        # Run detection and tracking
        clip = ClipDefinition(
            clip_id=request.clip_id,
            camera_ids=request.cameras,
            frames_per_camera=frames_per_camera,
            start_frame=0,
            end_frame=request.num_frames,
        )
        
        tracking_results_list = run_detection_and_tracking(clip)
        
        # Convert to dict format expected by build_frame_states
        tracking_results_dict = {}
        for camera_id in request.cameras:
            tracking_results_dict[camera_id] = tracking_results_list
        
        # Load camera calibrations
        calibration_path = Path(request.calibration_file)
        if not calibration_path.is_absolute():
            # Make path relative to project root
            project_root = Path(__file__).parent.parent
            calibration_path = project_root / calibration_path
        
        calibrations = load_calibration_from_file(calibration_path)
        
        # Use standard rugby field model
        field_model = get_standard_rugby_field()
        
        # Build 3D frame states
        frame_states = build_frame_states(
            tracking_results=tracking_results_dict,
            calibrations=calibrations,
            field_model=field_model,
            fps=30.0
        )
        
        # Get summary statistics
        summary = get_frame_state_summary(frame_states)
        
        # Format response
        response = {
            "clip_id": request.clip_id,
            "n_frames": len(frame_states),
            "summary": summary,
            "frame_states": [
                {
                    "frame_number": fs.frame_number,
                    "timestamp": fs.timestamp,
                    "ball_detected": fs.ball_detected,
                    "ball_position": fs.ball_pos_3d.tolist() if fs.ball_pos_3d is not None else None,
                    "n_players_tracked": fs.n_players_tracked,
                    "player_positions": {
                        int(track_id): pos.tolist()
                        for track_id, pos in fs.players_pos_3d.items()
                    }
                }
                for fs in frame_states
            ],
            "metadata": {
                "phase": "Phase 5 (3D Reconstruction)",
                "field_dimensions": {
                    "length_m": field_model.field_length,
                    "width_m": field_model.field_width,
                },
                "cameras_used": request.cameras,
                "calibration_file": str(calibration_path),
            },
        }
        
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Calibration file not found: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Calibration file not found: {str(e)}"
        )
    except ImportError as e:
        logger.error(f"Failed to import ml modules: {e}")
        raise HTTPException(
            status_code=500,
            detail="ML modules not available. Check system configuration."
        )
    except Exception as e:
        logger.error(f"3D reconstruction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"3D reconstruction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
