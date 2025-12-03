# Camera Calibration Configuration

This directory contains camera calibration files for the Rugby Vision system.

## File Format

Calibration files can be in JSON or YAML format and must follow this structure:

```json
{
  "cameras": {
    "camera_id": {
      "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "extrinsic": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]]
    }
  }
}
```

## Intrinsic Matrix

The intrinsic matrix (3x3) contains the camera's internal parameters:

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

- `fx`, `fy`: Focal lengths in pixels (X and Y directions)
- `cx`, `cy`: Principal point (optical center) in pixels

For a 1920x1080 camera, typical values:
- `fx`, `fy`: 1400-1600 (narrower FOV) or 800-1200 (wider FOV)
- `cx`: ~960 (half of image width)
- `cy`: ~540 (half of image height)

## Extrinsic Matrix

The extrinsic matrix (4x4) defines the transformation from world coordinates to camera coordinates:

```
[[r11, r12, r13, tx],
 [r21, r22, r23, ty],
 [r31, r32, r33, tz],
 [  0,   0,   0,  1]]
```

- Top-left 3x3: Rotation matrix (R) - must be orthogonal
- Top-right 3x1: Translation vector (t) - camera position in world coords
- Bottom row: Always [0, 0, 0, 1]

## Coordinate Systems

### World Coordinates (Rugby Field)
- **Origin**: Corner of try line (left corner when viewing from center)
- **X-axis**: Along touchline (0 to 70m)
- **Y-axis**: Along field length (0 to 100m)
- **Z-axis**: Vertical, upward (0 = ground level)

### Camera Coordinates (OpenCV Convention)
- **X-axis**: Right
- **Y-axis**: Down
- **Z-axis**: Forward (into the scene)

## Example Files

- `camera_calibration_example.json`: Synthetic 3-camera setup for testing

## Obtaining Real Calibration Data

For production use, camera calibration should be performed using:

1. **Intrinsic calibration**: Use OpenCV's `calibrateCamera()` with checkerboard patterns
2. **Extrinsic calibration**: Use known field points (corner flags, line intersections) with `solvePnP()`

See the Rugby Vision documentation for detailed calibration procedures.

## Validation

The system automatically validates:
- Matrix shapes (3x3 for intrinsic, 4x4 for extrinsic)
- Positive focal lengths
- Orthogonal rotation matrices
- Proper homogeneous coordinate format
