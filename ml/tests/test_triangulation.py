"""Tests for multi-view triangulation."""

import pytest
import numpy as np

from ml.calibration import CameraCalibration
from ml.triangulation import (
    triangulate_point,
    triangulate_points_batch,
    compute_reprojection_errors,
)


def create_test_calibration(camera_id: str, translation: np.ndarray) -> CameraCalibration:
    """Create a test camera calibration."""
    intrinsic = np.array([
        [1500.0, 0.0, 960.0],
        [0.0, 1500.0, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = translation
    
    return CameraCalibration(
        camera_id=camera_id,
        intrinsic=intrinsic,
        extrinsic=extrinsic
    )


class TestTriangulatePoint:
    """Tests for single point triangulation."""
    
    def test_simple_triangulation_2_cameras(self):
        """Test triangulation with 2 cameras and known point."""
        # Create two cameras separated along X-axis
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        
        # Point in world coordinates
        point_3d = np.array([1.0, 0.0, 5.0])
        
        # Project point into both cameras
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        obs_0 = project_point(cam_0, point_3d)
        obs_1 = project_point(cam_1, point_3d)
        
        observations = [(cam_0, obs_0), (cam_1, obs_1)]
        
        # Triangulate
        reconstructed = triangulate_point(observations)
        
        assert reconstructed is not None
        assert np.allclose(reconstructed, point_3d, atol=0.1)
    
    def test_triangulation_3_cameras(self):
        """Test triangulation with 3 cameras."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        cam_2 = create_test_calibration("cam_2", np.array([1.0, 2.0, 0.0]))
        
        point_3d = np.array([1.0, 1.0, 10.0])
        
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        observations = [
            (cam_0, project_point(cam_0, point_3d)),
            (cam_1, project_point(cam_1, point_3d)),
            (cam_2, project_point(cam_2, point_3d)),
        ]
        
        reconstructed = triangulate_point(observations)
        
        assert reconstructed is not None
        assert np.allclose(reconstructed, point_3d, atol=0.1)
    
    def test_insufficient_observations(self):
        """Test that single observation raises error."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        observations = [(cam_0, (640.0, 480.0))]
        
        with pytest.raises(ValueError, match="Need at least 2 observations"):
            triangulate_point(observations)
    
    def test_zero_observations(self):
        """Test that empty observations raises error."""
        with pytest.raises(ValueError, match="Need at least 2 observations"):
            triangulate_point([])
    
    def test_triangulation_with_noise(self):
        """Test triangulation with noisy observations."""
        np.random.seed(42)
        
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([3.0, 0.0, 0.0]))
        
        point_3d = np.array([1.5, 0.5, 8.0])
        
        def project_point_noisy(calib, point, noise_std=1.0):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            x = proj_h[0] / proj_h[2]
            y = proj_h[1] / proj_h[2]
            # Add Gaussian noise
            x += np.random.normal(0, noise_std)
            y += np.random.normal(0, noise_std)
            return x, y
        
        observations = [
            (cam_0, project_point_noisy(cam_0, point_3d)),
            (cam_1, project_point_noisy(cam_1, point_3d)),
        ]
        
        reconstructed = triangulate_point(observations)
        
        assert reconstructed is not None
        # With noise, allow larger tolerance
        assert np.allclose(reconstructed, point_3d, atol=0.5)


class TestTriangulatePointsBatch:
    """Tests for batch triangulation."""
    
    def test_batch_triangulation(self):
        """Test triangulating multiple points at once."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        
        points_3d = [
            np.array([1.0, 0.0, 5.0]),
            np.array([1.5, 1.0, 6.0]),
            np.array([0.5, -0.5, 7.0]),
        ]
        
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        observations_batch = [
            [
                (cam_0, project_point(cam_0, pt)),
                (cam_1, project_point(cam_1, pt)),
            ]
            for pt in points_3d
        ]
        
        reconstructed = triangulate_points_batch(observations_batch)
        
        assert len(reconstructed) == 3
        for i, rec in enumerate(reconstructed):
            assert rec is not None
            assert np.allclose(rec, points_3d[i], atol=0.1)
    
    def test_batch_with_failures(self):
        """Test batch triangulation with some failures."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        
        # Valid observations (2 cameras)
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        point_3d = np.array([1.0, 0.0, 5.0])
        
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        valid_obs = [
            (cam_0, project_point(cam_0, point_3d)),
            (cam_1, project_point(cam_1, point_3d)),
        ]
        
        # Invalid observations (only 1 camera)
        invalid_obs = [(cam_0, (640.0, 480.0))]
        
        observations_batch = [valid_obs, invalid_obs]
        
        # Should handle the error gracefully
        try:
            reconstructed = triangulate_points_batch(observations_batch)
            # First should succeed, second should fail
            assert reconstructed[0] is not None
        except ValueError:
            # Expected for invalid observations
            pass


class TestReprojectionErrors:
    """Tests for reprojection error computation."""
    
    def test_perfect_reprojection(self):
        """Test reprojection error for perfectly triangulated point."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        
        point_3d = np.array([1.0, 0.0, 5.0])
        
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        observations = [
            (cam_0, project_point(cam_0, point_3d)),
            (cam_1, project_point(cam_1, point_3d)),
        ]
        
        errors = compute_reprojection_errors(point_3d, observations)
        
        assert len(errors) == 2
        # Should have near-zero error
        assert all(e < 0.01 for e in errors)
    
    def test_reprojection_with_error(self):
        """Test reprojection error with intentional mismatch."""
        cam_0 = create_test_calibration("cam_0", np.array([0.0, 0.0, 0.0]))
        cam_1 = create_test_calibration("cam_1", np.array([2.0, 0.0, 0.0]))
        
        point_3d = np.array([1.0, 0.0, 5.0])
        
        def project_point(calib, point):
            P = calib.get_projection_matrix()
            point_h = np.append(point, 1.0)
            proj_h = P @ point_h
            return proj_h[0] / proj_h[2], proj_h[1] / proj_h[2]
        
        # Add offset to observations
        obs_0 = project_point(cam_0, point_3d)
        obs_1 = project_point(cam_1, point_3d)
        obs_1 = (obs_1[0] + 10.0, obs_1[1] + 5.0)  # Add error
        
        observations = [(cam_0, obs_0), (cam_1, obs_1)]
        
        errors = compute_reprojection_errors(point_3d, observations)
        
        assert len(errors) == 2
        # First should be near-zero, second should have error
        assert errors[0] < 0.01
        assert errors[1] > 5.0  # Should detect the added error
