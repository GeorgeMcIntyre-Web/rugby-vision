"""Tests for camera calibration module."""

import json
import pytest
import numpy as np
from pathlib import Path
import tempfile

from ml.calibration import (
    CameraCalibration,
    load_calibration_from_file,
    validate_calibration_set,
)


class TestCameraCalibration:
    """Tests for CameraCalibration dataclass."""
    
    def test_valid_calibration(self):
        """Test creating a valid calibration."""
        intrinsic = np.array([
            [1500.0, 0.0, 960.0],
            [0.0, 1500.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        extrinsic = np.eye(4)
        
        calibration = CameraCalibration(
            camera_id="cam_0",
            intrinsic=intrinsic,
            extrinsic=extrinsic
        )
        
        assert calibration.camera_id == "cam_0"
        assert np.array_equal(calibration.intrinsic, intrinsic)
        assert np.array_equal(calibration.extrinsic, extrinsic)
    
    def test_invalid_camera_id(self):
        """Test that empty camera_id raises error."""
        intrinsic = np.eye(3)
        extrinsic = np.eye(4)
        
        with pytest.raises(ValueError, match="camera_id must be a non-empty string"):
            CameraCalibration(
                camera_id="",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_invalid_intrinsic_shape(self):
        """Test that wrong intrinsic shape raises error."""
        intrinsic = np.eye(4)  # Wrong shape
        extrinsic = np.eye(4)
        
        with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_invalid_intrinsic_bottom_row(self):
        """Test that invalid intrinsic bottom row raises error."""
        intrinsic = np.array([
            [1500.0, 0.0, 960.0],
            [0.0, 1500.0, 540.0],
            [1.0, 0.0, 1.0]  # Invalid
        ])
        extrinsic = np.eye(4)
        
        with pytest.raises(ValueError, match="Intrinsic matrix bottom row must be"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_negative_focal_length(self):
        """Test that negative focal length raises error."""
        intrinsic = np.array([
            [-1500.0, 0.0, 960.0],  # Negative fx
            [0.0, 1500.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        extrinsic = np.eye(4)
        
        with pytest.raises(ValueError, match="Focal lengths must be positive"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_invalid_extrinsic_shape(self):
        """Test that wrong extrinsic shape raises error."""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1500.0
        intrinsic[1, 1] = 1500.0
        
        extrinsic = np.eye(3)  # Wrong shape
        
        with pytest.raises(ValueError, match="Extrinsic matrix must be 4x4"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_invalid_extrinsic_bottom_row(self):
        """Test that invalid extrinsic bottom row raises error."""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1500.0
        intrinsic[1, 1] = 1500.0
        
        extrinsic = np.eye(4)
        extrinsic[3, 0] = 1.0  # Invalid
        
        with pytest.raises(ValueError, match="Extrinsic matrix bottom row must be"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_non_orthogonal_rotation(self):
        """Test that non-orthogonal rotation matrix raises error."""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1500.0
        intrinsic[1, 1] = 1500.0
        
        extrinsic = np.eye(4)
        extrinsic[0, 0] = 2.0  # Makes rotation non-orthogonal
        
        with pytest.raises(ValueError, match="Extrinsic rotation matrix must be orthogonal"):
            CameraCalibration(
                camera_id="cam_0",
                intrinsic=intrinsic,
                extrinsic=extrinsic
            )
    
    def test_get_projection_matrix(self):
        """Test projection matrix computation."""
        intrinsic = np.array([
            [1500.0, 0.0, 960.0],
            [0.0, 1500.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        extrinsic = np.eye(4)
        
        calibration = CameraCalibration(
            camera_id="cam_0",
            intrinsic=intrinsic,
            extrinsic=extrinsic
        )
        
        P = calibration.get_projection_matrix()
        
        # Check shape
        assert P.shape == (3, 4)
        
        # For identity extrinsic, P should be K[I|0]
        expected = intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))])
        assert np.allclose(P, expected)


class TestLoadCalibration:
    """Tests for loading calibration from files."""
    
    def test_load_valid_json(self):
        """Test loading valid JSON calibration file."""
        calibration_data = {
            "cameras": {
                "cam_0": {
                    "intrinsic": [
                        [1500.0, 0.0, 960.0],
                        [0.0, 1500.0, 540.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "extrinsic": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(calibration_data, f)
            temp_path = f.name
        
        try:
            calibrations = load_calibration_from_file(temp_path)
            
            assert "cam_0" in calibrations
            assert calibrations["cam_0"].camera_id == "cam_0"
            assert calibrations["cam_0"].intrinsic[0, 0] == 1500.0
        finally:
            Path(temp_path).unlink()
    
    def test_load_multiple_cameras(self):
        """Test loading multiple cameras."""
        calibration_data = {
            "cameras": {
                "cam_0": {
                    "intrinsic": [[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]],
                    "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                },
                "cam_1": {
                    "intrinsic": [[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]],
                    "extrinsic": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(calibration_data, f)
            temp_path = f.name
        
        try:
            calibrations = load_calibration_from_file(temp_path)
            
            assert len(calibrations) == 2
            assert "cam_0" in calibrations
            assert "cam_1" in calibrations
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_calibration_from_file("/nonexistent/path.json")
    
    def test_missing_cameras_key(self):
        """Test that missing 'cameras' key raises error."""
        data = {"invalid": {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must contain 'cameras' key"):
                load_calibration_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_missing_intrinsic(self):
        """Test that missing intrinsic matrix raises error."""
        data = {
            "cameras": {
                "cam_0": {
                    "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing 'intrinsic' matrix"):
                load_calibration_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestValidateCalibrationSet:
    """Tests for calibration set validation."""
    
    def test_valid_set(self):
        """Test validating a valid calibration set."""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1500.0
        intrinsic[1, 1] = 1500.0
        
        calibrations = {
            "cam_0": CameraCalibration("cam_0", intrinsic, np.eye(4)),
            "cam_1": CameraCalibration("cam_1", intrinsic, np.eye(4)),
        }
        
        # Should not raise
        validate_calibration_set(calibrations)
    
    def test_insufficient_cameras(self):
        """Test that single camera raises error."""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 1500.0
        intrinsic[1, 1] = 1500.0
        
        calibrations = {
            "cam_0": CameraCalibration("cam_0", intrinsic, np.eye(4)),
        }
        
        with pytest.raises(ValueError, match="Need at least 2 cameras"):
            validate_calibration_set(calibrations)
