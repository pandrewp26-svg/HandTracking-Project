"""Convert MediaPipe landmarks into hand measurements.

MediaPipe gives 21 normalized landmark positions.  The helpers below
change those positions into pixels, joint angles, and relative depth samples.
"""

import numpy as np


FINGERS = {
    "Thumb": (2, 3, 4),
    "Index": (5, 6, 7),
    "Middle": (9, 10, 11),
    "Ring": (13, 14, 15),
    "Pinky": (17, 18, 19),
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

AXES = ("x", "y", "z")


def landmarks_to_pixels(landmarks, frame_shape):
    """Convert image landmark positions to pixel coordinates for drawing."""
    height, width = frame_shape[:2]
    return np.array(
        [[landmark.x * width, landmark.y * height] for landmark in landmarks]
    )


def world_landmarks_to_points(world_landmarks):
    """Convert MediaPipe world landmarks into a 21 by 3 coordinate array."""
    if isinstance(world_landmarks, np.ndarray):
        return world_landmarks
    return np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in world_landmarks
    ])


def sample_depth_neighborhood(
    depth_map, landmarks, radius=3, return_validity=False
):
    """Sample finite depth values in a clipped square neighborhood per landmark."""
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a two-dimensional array")
    if not depth_map.size:
        raise ValueError("depth_map must not be empty")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    height, width = depth_map.shape
    values = np.full(len(landmarks), np.nan, dtype=float)
    validity = np.zeros(len(landmarks), dtype=bool)
    for index, landmark in enumerate(landmarks):
        if hasattr(landmark, "x"):
            normalized_x, normalized_y = landmark.x, landmark.y
        else:
            normalized_x, normalized_y = landmark[:2]
        if not np.isfinite(normalized_x) or not np.isfinite(normalized_y):
            continue
        x = int(round(normalized_x * (width - 1)))
        y = int(round(normalized_y * (height - 1)))
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        patch = depth_map[
            max(0, y - radius):min(height, y + radius + 1),
            max(0, x - radius):min(width, x + radius + 1),
        ]
        finite_values = patch[np.isfinite(patch)]
        if finite_values.size:
            values[index] = float(np.median(finite_values))
            validity[index] = True
    if return_validity:
        return values, validity
    return values


def depth_relative_coordinates(landmarks, relative_depths):
    """Build normalized-image XY plus model-relative Z, not metric camera XYZ."""
    relative_depths = np.asarray(relative_depths, dtype=float)
    if relative_depths.shape != (len(landmarks),):
        raise ValueError("relative_depths must contain one value per landmark")
    normalized_points = [
        (landmark.x, landmark.y) if hasattr(landmark, "x") else landmark[:2]
        for landmark in landmarks
    ]
    return np.array([
        [x, y, depth]
        for (x, y), depth in zip(normalized_points, relative_depths)
    ])


def bend_angle_world(world_points, a, b, c):
    """3D joint bend using MediaPipe world-landmark coordinates.

    A straight arrangement returns 0 degrees.
    """
    world_points = np.asarray(world_points, dtype=float)
    v1 = world_points[a] - world_points[b]
    v2 = world_points[c] - world_points[b]
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if not np.isfinite(denominator) or denominator < 1e-8:
        return None

    cosine = np.clip(np.dot(v1, v2) / denominator, -1, 1)
    return 180 - np.degrees(np.arccos(cosine))


def palm_orientation(points):
    """2D wrist-to-middle-MCP direction."""
    v = points[9] - points[0]
    return np.degrees(np.arctan2(v[1], v[0]))


def palm_normal(points):
    """Approximate 3D normal from wrist, index MCP, and pinky MCP."""
    wrist = points[0]
    # The cross product uses the same index-minus-wrist and pinky-minus-wrist
    # convention for image-derived and world-landmark coordinates.
    normal = np.cross(points[5] - wrist, points[17] - wrist)
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-8 else None
