"""Rugby field coordinate system and transformations.

Defines the rugby field reference frame and provides utilities for
transforming 3D points from world coordinates to field coordinates.

Rugby Field Coordinate System:
    - Origin: Corner of the try line (left corner when viewing from center)
    - X-axis: Along the touchline (0 to field_width, typically 70m)
    - Y-axis: Along the field length (0 to field_length, typically 100m)
    - Z-axis: Vertical, pointing upward (0 = ground level)

Standard Rugby Field Dimensions (World Rugby regulations):
    - Length: 100m (between try lines)
    - Width: 70m (between touchlines)
    - Try line to dead ball line: 10-22m (in-goal area)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FieldModel:
    """Rugby field dimensions and coordinate system definition.
    
    Attributes:
        field_length: Length of field in meters (try line to try line)
        field_width: Width of field in meters (touchline to touchline)
        try_line_0: Y-coordinate of first try line (typically 0)
        try_line_1: Y-coordinate of second try line (typically field_length)
        origin_offset: 3D offset from world origin to field origin [X, Y, Z]
        rotation_matrix: 3x3 rotation from world to field coordinates
    """
    field_length: float = 100.0
    field_width: float = 70.0
    try_line_0: float = 0.0
    try_line_1: float = 100.0
    origin_offset: np.ndarray = None
    rotation_matrix: np.ndarray = None
    
    def __post_init__(self) -> None:
        """Initialize default values and validate."""
        if self.origin_offset is None:
            self.origin_offset = np.zeros(3, dtype=np.float64)
        
        if self.rotation_matrix is None:
            self.rotation_matrix = np.eye(3, dtype=np.float64)
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate field model parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if self.field_length <= 0:
            raise ValueError(f"field_length must be positive, got {self.field_length}")
        
        if self.field_width <= 0:
            raise ValueError(f"field_width must be positive, got {self.field_width}")
        
        if self.try_line_1 <= self.try_line_0:
            raise ValueError("try_line_1 must be greater than try_line_0")
        
        if self.origin_offset.shape != (3,):
            raise ValueError(f"origin_offset must be 3D vector, got shape {self.origin_offset.shape}")
        
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix must be 3x3, got shape {self.rotation_matrix.shape}")
        
        # Check rotation matrix is orthogonal
        identity_check = self.rotation_matrix @ self.rotation_matrix.T
        if not np.allclose(identity_check, np.eye(3), atol=1e-3):
            raise ValueError("rotation_matrix must be orthogonal")


def transform_to_field_coords(
    point_3d: np.ndarray,
    field_model: FieldModel
) -> Optional[np.ndarray]:
    """Transform 3D world point to rugby field coordinate system.
    
    Args:
        point_3d: 3D point in world coordinates [X, Y, Z]
        field_model: Rugby field model with transformation parameters
    
    Returns:
        3D point in field coordinates [X_field, Y_field, Z_field] or None
        if point is invalid (e.g., below ground)
    """
    if point_3d.shape != (3,):
        raise ValueError(f"point_3d must be 3D vector, got shape {point_3d.shape}")
    
    # Apply transformation: R * (p - offset)
    centered = point_3d - field_model.origin_offset
    field_point = field_model.rotation_matrix @ centered
    
    # Validate Z coordinate (must be at or above ground)
    if field_point[2] < -0.1:  # Small tolerance for numerical errors
        return None
    
    # Clamp Z to ground level if slightly negative
    if field_point[2] < 0:
        field_point[2] = 0.0
    
    return field_point


def is_point_in_field_bounds(
    field_point: np.ndarray,
    field_model: FieldModel,
    tolerance: float = 5.0
) -> bool:
    """Check if point is within field bounds with tolerance.
    
    Args:
        field_point: 3D point in field coordinates
        field_model: Rugby field model
        tolerance: Boundary tolerance in meters (for out-of-bounds plays)
    
    Returns:
        True if point is within bounds (including tolerance)
    """
    if field_point.shape != (3,):
        return False
    
    x, y, z = field_point
    
    # Check Z (vertical) - must be reasonable height
    if z < -tolerance or z > 50.0:  # 50m is unreasonably high
        return False
    
    # Check X (width) with tolerance
    if x < -tolerance or x > field_model.field_width + tolerance:
        return False
    
    # Check Y (length) with tolerance
    if y < field_model.try_line_0 - tolerance:
        return False
    if y > field_model.try_line_1 + tolerance:
        return False
    
    return True


def get_standard_rugby_field() -> FieldModel:
    """Create a standard rugby field model with World Rugby dimensions.
    
    Returns:
        FieldModel with standard 100m x 70m dimensions
    """
    return FieldModel(
        field_length=100.0,
        field_width=70.0,
        try_line_0=0.0,
        try_line_1=100.0,
        origin_offset=np.zeros(3),
        rotation_matrix=np.eye(3)
    )


def compute_distance_2d(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """Compute 2D distance between two field points (ignoring Z).
    
    Args:
        point1: First 3D point in field coordinates
        point2: Second 3D point in field coordinates
    
    Returns:
        Distance in meters (2D ground plane)
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def compute_distance_3d(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """Compute 3D Euclidean distance between two field points.
    
    Args:
        point1: First 3D point in field coordinates
        point2: Second 3D point in field coordinates
    
    Returns:
        Distance in meters (3D)
    """
    return np.linalg.norm(point1 - point2)
