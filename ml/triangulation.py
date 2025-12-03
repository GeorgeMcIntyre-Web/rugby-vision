"""Multi-view triangulation for 3D point reconstruction.

Implements Direct Linear Transform (DLT) method for triangulating 3D points
from 2D observations across multiple calibrated cameras.

References:
    - Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
    - Chapter 12: Structure Computation
"""

from typing import List, Tuple, Optional

import numpy as np

from ml.calibration import CameraCalibration


Observation = Tuple[CameraCalibration, Tuple[float, float]]


def triangulate_point(observations: List[Observation]) -> Optional[np.ndarray]:
    """Triangulate a 3D point from multi-view 2D observations using DLT.
    
    Uses the Direct Linear Transform method to solve for the 3D point that
    best satisfies all 2D observations across multiple cameras.
    
    Args:
        observations: List of (CameraCalibration, (x, y)) tuples where
                     (x, y) are pixel coordinates in each camera view
    
    Returns:
        3D point as numpy array [X, Y, Z] in world coordinates, or None if
        triangulation fails
    
    Raises:
        ValueError: If fewer than 2 observations provided
    """
    if len(observations) < 2:
        raise ValueError(f"Need at least 2 observations for triangulation, got {len(observations)}")
    
    # Build the linear system A * X = 0 where X = [X, Y, Z, 1]^T
    A = []
    
    for calibration, (x, y) in observations:
        P = calibration.get_projection_matrix()  # 3x4
        
        # Each observation contributes 2 equations:
        # x * P[2,:] - P[0,:] = 0
        # y * P[2,:] - P[1,:] = 0
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    A = np.array(A)  # (2*n_cameras, 4)
    
    # Check for degenerate geometry
    if np.linalg.matrix_rank(A) < 3:
        return None  # Degenerate configuration
    
    # Solve using SVD: A * X = 0
    # Solution is the right singular vector corresponding to smallest singular value
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1, :]  # Last row of V^T
    
    # Convert from homogeneous to 3D coordinates
    if np.abs(X_homogeneous[3]) < 1e-10:
        return None  # Point at infinity
    
    point_3d = X_homogeneous[:3] / X_homogeneous[3]
    
    # Validate result: compute reprojection error
    max_error = _compute_max_reprojection_error(point_3d, observations)
    if max_error > 100.0:  # Threshold in pixels
        return None  # Likely incorrect triangulation
    
    return point_3d


def triangulate_points_batch(
    observations_batch: List[List[Observation]]
) -> List[Optional[np.ndarray]]:
    """Triangulate multiple 3D points from batch of observations.
    
    Args:
        observations_batch: List of observation lists, one per point
    
    Returns:
        List of 3D points (or None for failed triangulations)
    """
    return [triangulate_point(obs) for obs in observations_batch]


def _compute_max_reprojection_error(
    point_3d: np.ndarray,
    observations: List[Observation]
) -> float:
    """Compute maximum reprojection error across all observations.
    
    Args:
        point_3d: 3D point in world coordinates
        observations: List of (CameraCalibration, (x, y)) tuples
    
    Returns:
        Maximum reprojection error in pixels
    """
    max_error = 0.0
    point_homogeneous = np.append(point_3d, 1.0)  # [X, Y, Z, 1]
    
    for calibration, (x_obs, y_obs) in observations:
        P = calibration.get_projection_matrix()
        projected_homogeneous = P @ point_homogeneous
        
        # Convert to pixel coordinates
        if np.abs(projected_homogeneous[2]) < 1e-10:
            return float('inf')  # Behind camera
        
        x_proj = projected_homogeneous[0] / projected_homogeneous[2]
        y_proj = projected_homogeneous[1] / projected_homogeneous[2]
        
        error = np.sqrt((x_proj - x_obs)**2 + (y_proj - y_obs)**2)
        max_error = max(max_error, error)
    
    return max_error


def compute_reprojection_errors(
    point_3d: np.ndarray,
    observations: List[Observation]
) -> List[float]:
    """Compute reprojection error for each observation.
    
    Args:
        point_3d: 3D point in world coordinates
        observations: List of (CameraCalibration, (x, y)) tuples
    
    Returns:
        List of reprojection errors in pixels, one per observation
    """
    errors = []
    point_homogeneous = np.append(point_3d, 1.0)
    
    for calibration, (x_obs, y_obs) in observations:
        P = calibration.get_projection_matrix()
        projected_homogeneous = P @ point_homogeneous
        
        if np.abs(projected_homogeneous[2]) < 1e-10:
            errors.append(float('inf'))
            continue
        
        x_proj = projected_homogeneous[0] / projected_homogeneous[2]
        y_proj = projected_homogeneous[1] / projected_homogeneous[2]
        
        error = np.sqrt((x_proj - x_obs)**2 + (y_proj - y_obs)**2)
        errors.append(error)
    
    return errors
