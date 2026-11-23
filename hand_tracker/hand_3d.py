"""Turn hand landmarks into display coordinates.

These coordinates are only for drawing a hand-shaped 3D preview.
They are not calibrated physical coordinates.
"""

import numpy as np


LANDMARK_COUNT = 21
DEFAULT_DEPTH_SCALE = 0.8

FINGER_CONNECTIONS = {
    "Thumb": ((0, 1), (1, 2), (2, 3), (3, 4)),
    "Index": ((5, 6), (6, 7), (7, 8)),
    "Middle": ((9, 10), (10, 11), (11, 12)),
    "Ring": ((13, 14), (14, 15), (15, 16)),
    "Pinky": ((17, 18), (18, 19), (19, 20)),
}
PALM_CONNECTIONS = ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
CONNECTIONS = tuple(
    edge for edges in FINGER_CONNECTIONS.values() for edge in edges
) + PALM_CONNECTIONS


def build_hand_points(
    landmarks,
    relative_depths,
    depth_validity,
    image_shape,
    depth_scale=DEFAULT_DEPTH_SCALE,
):
    """Map normalized image coordinates and relative depth to display XYZ.

    X points right, Y points up, and positive Z follows larger DAV2 scores.
    Scores are robustly normalized per hand and are not metric depth.
    """
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks")
    depths = np.asarray(relative_depths, dtype=float)
    validity = np.asarray(depth_validity, dtype=bool)
    if depths.shape != (LANDMARK_COUNT,) or validity.shape != (LANDMARK_COUNT,):
        raise ValueError("Depth values and validity must have 21 entries")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain image height and width")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if not np.isfinite(depth_scale) or depth_scale < 0:
        raise ValueError("depth_scale must be a finite non-negative value")

    normalized_xy = np.asarray(
        [[landmark[0], landmark[1]] for landmark in landmarks],
        dtype=float,
    )
    if normalized_xy.shape != (LANDMARK_COUNT, 2):
        raise ValueError("Each landmark must provide normalized X and Y")
    finite_xy = np.isfinite(normalized_xy).all(axis=1)
    valid = validity & np.isfinite(depths) & finite_xy
    normalized_xy[~finite_xy] = 0.5

    short_side = float(min(height, width))
    points = np.zeros((LANDMARK_COUNT, 3), dtype=float)
    points[:, 0] = (normalized_xy[:, 0] - 0.5) * (width / short_side)
    points[:, 1] = (0.5 - normalized_xy[:, 1]) * (height / short_side)

    if valid.any():
        valid_depths = depths[valid]
        center = float(np.median(valid_depths))
        low, high = np.percentile(valid_depths, [10, 90])
        spread = float(high - low)
        if spread > 1e-8:
            points[valid, 2] = np.clip(
                (valid_depths - center) / spread,
                -1.0,
                1.0,
            ) * depth_scale

    return points, valid
