"""Tests for rugby field coordinate system."""

import pytest
import numpy as np

from ml.field_coords import (
    FieldModel,
    transform_to_field_coords,
    is_point_in_field_bounds,
    get_standard_rugby_field,
    compute_distance_2d,
    compute_distance_3d,
)


class TestFieldModel:
    """Tests for FieldModel dataclass."""
    
    def test_default_field(self):
        """Test creating field with default values."""
        field = FieldModel()
        
        assert field.field_length == 100.0
        assert field.field_width == 70.0
        assert field.try_line_0 == 0.0
        assert field.try_line_1 == 100.0
        assert np.array_equal(field.origin_offset, np.zeros(3))
        assert np.array_equal(field.rotation_matrix, np.eye(3))
    
    def test_custom_field(self):
        """Test creating field with custom dimensions."""
        field = FieldModel(
            field_length=120.0,
            field_width=80.0,
            try_line_0=10.0,
            try_line_1=110.0
        )
        
        assert field.field_length == 120.0
        assert field.field_width == 80.0
    
    def test_invalid_length(self):
        """Test that negative length raises error."""
        with pytest.raises(ValueError, match="field_length must be positive"):
            FieldModel(field_length=-10.0)
    
    def test_invalid_width(self):
        """Test that negative width raises error."""
        with pytest.raises(ValueError, match="field_width must be positive"):
            FieldModel(field_width=0.0)
    
    def test_invalid_try_lines(self):
        """Test that invalid try line positions raise error."""
        with pytest.raises(ValueError, match="try_line_1 must be greater than try_line_0"):
            FieldModel(try_line_0=100.0, try_line_1=50.0)
    
    def test_invalid_offset_shape(self):
        """Test that wrong offset shape raises error."""
        with pytest.raises(ValueError, match="origin_offset must be 3D vector"):
            FieldModel(origin_offset=np.array([0.0, 0.0]))
    
    def test_invalid_rotation_shape(self):
        """Test that wrong rotation shape raises error."""
        with pytest.raises(ValueError, match="rotation_matrix must be 3x3"):
            FieldModel(rotation_matrix=np.eye(4))
    
    def test_non_orthogonal_rotation(self):
        """Test that non-orthogonal rotation raises error."""
        rotation = np.array([[2.0, 0, 0], [0, 1, 0], [0, 0, 1]])  # Not orthogonal
        
        with pytest.raises(ValueError, match="rotation_matrix must be orthogonal"):
            FieldModel(rotation_matrix=rotation)


class TestTransformToFieldCoords:
    """Tests for coordinate transformation."""
    
    def test_identity_transform(self):
        """Test transform with identity rotation and zero offset."""
        field = FieldModel()
        point = np.array([35.0, 50.0, 1.5])
        
        transformed = transform_to_field_coords(point, field)
        
        assert transformed is not None
        assert np.allclose(transformed, point)
    
    def test_translation_only(self):
        """Test transform with translation offset."""
        offset = np.array([10.0, 20.0, 0.0])
        field = FieldModel(origin_offset=offset)
        
        point = np.array([45.0, 70.0, 2.0])
        transformed = transform_to_field_coords(point, field)
        
        expected = point - offset
        assert transformed is not None
        assert np.allclose(transformed, expected)
    
    def test_rotation_only(self):
        """Test transform with rotation."""
        # 90-degree rotation around Z-axis
        theta = np.pi / 2
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        field = FieldModel(rotation_matrix=rotation)
        
        point = np.array([1.0, 0.0, 0.0])
        transformed = transform_to_field_coords(point, field)
        
        # After 90-degree rotation, X -> Y
        expected = np.array([0.0, 1.0, 0.0])
        assert transformed is not None
        assert np.allclose(transformed, expected, atol=1e-10)
    
    def test_point_below_ground(self):
        """Test that point below ground returns None."""
        field = FieldModel()
        point = np.array([35.0, 50.0, -2.0])
        
        transformed = transform_to_field_coords(point, field)
        
        assert transformed is None
    
    def test_point_slightly_below_ground(self):
        """Test that point slightly below ground is clamped to zero."""
        field = FieldModel()
        point = np.array([35.0, 50.0, -0.01])
        
        transformed = transform_to_field_coords(point, field)
        
        assert transformed is not None
        assert transformed[2] == 0.0
        assert transformed[0] == 35.0
        assert transformed[1] == 50.0
    
    def test_invalid_point_shape(self):
        """Test that wrong point shape raises error."""
        field = FieldModel()
        point = np.array([35.0, 50.0])  # 2D instead of 3D
        
        with pytest.raises(ValueError, match="point_3d must be 3D vector"):
            transform_to_field_coords(point, field)


class TestIsPointInFieldBounds:
    """Tests for field bounds checking."""
    
    def test_point_inside_bounds(self):
        """Test point clearly inside field."""
        field = FieldModel()
        point = np.array([35.0, 50.0, 1.0])
        
        assert is_point_in_field_bounds(point, field)
    
    def test_point_outside_x_bounds(self):
        """Test point outside X (width) bounds."""
        field = FieldModel()
        point = np.array([100.0, 50.0, 1.0])  # X > field_width
        
        assert not is_point_in_field_bounds(point, field)
    
    def test_point_outside_y_bounds(self):
        """Test point outside Y (length) bounds."""
        field = FieldModel()
        point = np.array([35.0, 150.0, 1.0])  # Y > try_line_1
        
        assert not is_point_in_field_bounds(point, field)
    
    def test_point_outside_z_bounds(self):
        """Test point outside Z (height) bounds."""
        field = FieldModel()
        point = np.array([35.0, 50.0, 100.0])  # Z too high
        
        assert not is_point_in_field_bounds(point, field)
    
    def test_point_with_tolerance(self):
        """Test point outside bounds but within tolerance."""
        field = FieldModel()
        point = np.array([72.0, 50.0, 1.0])  # X slightly > field_width (70)
        
        # Should fail with default tolerance
        assert not is_point_in_field_bounds(point, field, tolerance=1.0)
        
        # Should pass with larger tolerance
        assert is_point_in_field_bounds(point, field, tolerance=5.0)
    
    def test_point_on_boundary(self):
        """Test point exactly on field boundary."""
        field = FieldModel()
        point = np.array([70.0, 100.0, 0.0])  # On corner
        
        assert is_point_in_field_bounds(point, field)
    
    def test_invalid_point_shape(self):
        """Test that wrong point shape returns False."""
        field = FieldModel()
        point = np.array([35.0, 50.0])  # 2D
        
        assert not is_point_in_field_bounds(point, field)


class TestGetStandardRugbyField:
    """Tests for standard field factory."""
    
    def test_standard_dimensions(self):
        """Test standard rugby field has correct dimensions."""
        field = get_standard_rugby_field()
        
        assert field.field_length == 100.0
        assert field.field_width == 70.0
        assert field.try_line_0 == 0.0
        assert field.try_line_1 == 100.0


class TestDistanceComputation:
    """Tests for distance computation functions."""
    
    def test_distance_2d(self):
        """Test 2D distance computation."""
        point1 = np.array([0.0, 0.0, 5.0])
        point2 = np.array([3.0, 4.0, 10.0])  # Z difference ignored
        
        distance = compute_distance_2d(point1, point2)
        
        # 3-4-5 triangle
        assert np.isclose(distance, 5.0)
    
    def test_distance_2d_same_point(self):
        """Test 2D distance for same point."""
        point = np.array([1.0, 2.0, 3.0])
        
        distance = compute_distance_2d(point, point)
        
        assert np.isclose(distance, 0.0)
    
    def test_distance_3d(self):
        """Test 3D distance computation."""
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([1.0, 2.0, 2.0])
        
        distance = compute_distance_3d(point1, point2)
        
        # sqrt(1 + 4 + 4) = 3
        assert np.isclose(distance, 3.0)
    
    def test_distance_3d_same_point(self):
        """Test 3D distance for same point."""
        point = np.array([1.0, 2.0, 3.0])
        
        distance = compute_distance_3d(point, point)
        
        assert np.isclose(distance, 0.0)
