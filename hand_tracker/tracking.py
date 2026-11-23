"""Prepare MediaPipe and read 21 hand points from one frame."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_hand_landmarker(model_path):
    """Create a one-hand MediaPipe tracker from the hand model file."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_landmarks(tracker, frame, timestamp_ms):
    """Find image and world landmark sets for the tracked hand in one frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = tracker.detect_for_video(image, timestamp_ms)

    image_hands = getattr(result, "hand_landmarks", ())
    world_hands = getattr(result, "hand_world_landmarks", ())
    handedness_hands = getattr(result, "handedness", ())

    # World and image landmarks must describe the same hand.
    if not image_hands or not world_hands:
        return (), (), ()
    if len(image_hands) != len(world_hands):
        return (), (), ()
    if len(image_hands[0]) != 21 or len(world_hands[0]) != 21:
        return (), (), ()

    handedness = ()
    if handedness_hands:
        first = handedness_hands[0]
        if first:
            label = getattr(first[0], "category_name", "")
            score = getattr(first[0], "score", 0.0)
            handedness = (str(label), float(score))

    return image_hands, world_hands, handedness
