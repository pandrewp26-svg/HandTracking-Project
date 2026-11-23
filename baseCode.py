"""Simple hand-angle recorder using a webcam and MediaPipe.

This is the beginner-friendly, no-depth version of the tracker.
It is meant to be easy to read: setup happens first, then one loop
processes each camera frame.

Keys:
    q or Escape  - quit
    r            - reset the angle filters
    p            - save plots
"""
print('w')
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------------------------------------------
# File and measurement settings
# ----------------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_FOLDER / "hand_landmarker.task"
OUTPUT_FOLDER = PROJECT_FOLDER / "hand_tracker_data"
PLOT_FOLDER = OUTPUT_FOLDER / "plots"

TRIAL_GAP_SECONDS = 2.0
MAX_PLOT_POINTS = 300

FINGERS = {
    "Thumb": (2, 3, 4),
    "Index": (5, 6, 7),
    "Middle": (9, 10, 11),
    "Ring": (13, 14, 15),
    "Pinky": (17, 18, 19),
}

HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

AXES = ("x", "y", "z")


# ----------------------------------------------------------------------
# Hand measurements
# ----------------------------------------------------------------------

def joint_bend_angle(points, first, middle, last):
    """Return the bend at a finger joint.

    A straight finger is close to 0 degrees.
    """
    from_middle_to_first = points[first] - points[middle]
    from_middle_to_last = points[last] - points[middle]

    lengths = np.linalg.norm(from_middle_to_first) * np.linalg.norm(from_middle_to_last)
    if lengths < 1e-8:
        return None

    cosine = np.dot(from_middle_to_first, from_middle_to_last) / lengths
    cosine = np.clip(cosine, -1, 1)
    return 180 - np.degrees(np.arccos(cosine))


def palm_direction_2d(points):
    """Return the wrist-to-middle-finger direction in degrees."""
    direction = points[9] - points[0]
    return np.degrees(np.arctan2(direction[1], direction[0]))


def palm_normal_3d(points):
    """Estimate the direction perpendicular to the palm."""
    wrist = points[0]
    index_direction = points[5] - wrist
    pinky_direction = points[17] - wrist
    normal = np.cross(index_direction, pinky_direction)

    length = np.linalg.norm(normal)
    if length < 1e-8:
        return None
    return normal / length


# ----------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------

class AngleFilter:
    """Smooth one angle using a small Kalman filter."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.estimated_state = None
        self.error_matrix = np.diag([100.0, 400.0])
        self.previous_time = None

    def update(self, new_angle, now):
        if self.estimated_state is None:
            self.estimated_state = np.array([new_angle, 0.0])
            self.previous_time = now
            return new_angle

        time_step = now - self.previous_time
        time_step = np.clip(time_step, 0.001, 0.25)
        self.previous_time = now

        motion = np.array([[1, time_step], [0, 1]], dtype=float)
        process_noise = 0.8 * np.array([
            [time_step ** 4 / 4, time_step ** 3 / 2],
            [time_step ** 3 / 2, time_step ** 2],
        ])

        self.estimated_state = motion @ self.estimated_state
        self.error_matrix = motion @ self.error_matrix @ motion.T + process_noise

        measurement = np.array([[1.0, 0.0]])
        measurement_noise = np.array([[4.0]])
        difference = np.array([new_angle]) - measurement @ self.estimated_state
        uncertainty = measurement @ self.error_matrix @ measurement.T + measurement_noise
        gain = self.error_matrix @ measurement.T @ np.linalg.inv(uncertainty)

        self.estimated_state += (gain @ difference).ravel()
        identity = np.eye(2)
        self.error_matrix = (identity - gain @ measurement) @ self.error_matrix
        return float(self.estimated_state[0])


# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------

def get_next_session_number():
    if not OUTPUT_FOLDER.exists():
        return 1

    session_numbers = []
    for csv_file in OUTPUT_FOLDER.glob("session_*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            session_numbers.append(int(parts[1]))

    if session_numbers:
        return max(session_numbers) + 1
    return 1


def make_csv_columns():
    columns = [
        "timestamp", "elapsed_s", "frame", "trial", "hand_detected",
        "hand_absent_s", "palm_orientation_2d_deg", "landmarks_2d_json",
    ]

    for axis in AXES:
        columns.append(f"palm_normal_{axis}")
        columns.append(f"palm_normal_{axis}_angle_raw_deg")
        columns.append(f"palm_normal_{axis}_angle_filtered_deg")

    for finger_name in FINGERS:
        columns.append(f"{finger_name.lower()}_raw_deg")
        columns.append(f"{finger_name.lower()}_filtered_deg")
    return columns


def save_plots(bend_history, normal_history, session_number, timestamp):
    if bend_history:
        data = np.array(bend_history)
        plt.figure(figsize=(10, 4))
        plt.plot(data[:, 0], data[:, 1], label="Raw")
        plt.plot(data[:, 0], data[:, 2], label="Filtered")
        plt.xlabel("Frame")
        plt.ylabel("Index bend (degrees)")
        plt.title("Raw vs Filtered 2D Index Bend")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDER / f"session_{session_number:03d}_{timestamp}_bend.png",
            dpi=150,
        )
        plt.close()

    if normal_history:
        data = np.array(normal_history)
        plt.figure(figsize=(10, 5))
        for column, axis in enumerate(AXES, start=1):
            plt.plot(data[:, 0], data[:, column], label=f"{axis.upper()} angle")
        plt.xlabel("Frame")
        plt.ylabel("Angle to positive axis (degrees)")
        plt.title("Approximate Palm-Normal Axis Angles")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDER / f"session_{session_number:03d}_{timestamp}_normal.png",
            dpi=150,
        )
        plt.close()


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    session_number = get_next_session_number()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_FOLDER / f"session_{session_number:03d}_{timestamp}.csv"
    csv_columns = make_csv_columns()

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    finger_filters = {name: AngleFilter() for name in FINGERS}
    normal_filters = {axis: AngleFilter() for axis in AXES}
    bend_history = deque(maxlen=MAX_PLOT_POINTS)
    normal_history = deque(maxlen=MAX_PLOT_POINTS)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try camera index 1.")

    start_time = time.monotonic()
    last_hand_time = None
    frame_number = 0
    trial_number = 0
    last_timestamp_ms = -1

    try:
        with vision.HandLandmarker.create_from_options(hand_options) as tracker:
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                writer.writeheader()

                while True:
                    camera_read_ok, frame = camera.read()
                    if not camera_read_ok:
                        break

                    now = time.monotonic()
                    elapsed_seconds = now - start_time
                    frame_number += 1

                    timestamp_ms = int(elapsed_seconds * 1000)
                    timestamp_ms = max(last_timestamp_ms + 1, timestamp_ms)
                    last_timestamp_ms = timestamp_ms

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    detection = tracker.detect_for_video(image, timestamp_ms)

                    hand_was_detected = bool(detection.hand_landmarks)
                    row = {column: "" for column in csv_columns}
                    row["timestamp"] = time.time()
                    row["elapsed_s"] = round(elapsed_seconds, 4)
                    row["frame"] = frame_number
                    row["trial"] = trial_number
                    row["hand_detected"] = int(hand_was_detected)

                    if hand_was_detected:
                        # A new trial starts after the hand has been absent.
                        if last_hand_time is None or now - last_hand_time > TRIAL_GAP_SECONDS:
                            trial_number += 1
                            row["trial"] = trial_number
                            for angle_filter in (
                                *finger_filters.values(),
                                *normal_filters.values(),
                            ):
                                angle_filter.reset()
                            print(f"Trial {trial_number} started")

                        process_detected_hand(
                            frame,
                            detection.hand_landmarks[0],
                            now,
                            row,
                            frame_number,
                            trial_number,
                            last_hand_time,
                            finger_filters,
                            normal_filters,
                            bend_history,
                            normal_history,
                        )
                        last_hand_time = now

                    else:
                        if last_hand_time is not None:
                            absent_seconds = now - last_hand_time
                            row["hand_absent_s"] = round(absent_seconds, 3)
                            if absent_seconds > TRIAL_GAP_SECONDS:
                                last_hand_time = None

                    writer.writerow(row)
                    csv_file.flush()

                    cv2.putText(
                        frame,
                        f"Trial: {trial_number} | Frame: {frame_number}",
                        (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Hand Tracker", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        for angle_filter in (
                            *finger_filters.values(),
                            *normal_filters.values(),
                        ):
                            angle_filter.reset()
                        print("Filters reset.")
                    elif key == ord("p"):
                        save_plots(
                            bend_history,
                            normal_history,
                            session_number,
                            timestamp,
                        )
                        print(f"Plots saved in {PLOT_FOLDER}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    save_plots(bend_history, normal_history, session_number, timestamp)
    print(f"CSV saved: {csv_path}")
    print(f"Plots saved in: {PLOT_FOLDER}")


def process_detected_hand(
    frame,
    landmarks,
    now,
    row,
    frame_number,
    trial_number,
    last_hand_time,
    finger_filters,
    normal_filters,
    bend_history,
    normal_history,
):
    """Calculate and draw measurements for one detected hand."""
    del trial_number, last_hand_time
    frame_height, frame_width = frame.shape[:2]

    points_2d = np.array([
        [landmark.x * frame_width, landmark.y * frame_height]
        for landmark in landmarks
    ])
    points_3d = np.array([
        [landmark.x * frame_width, landmark.y * frame_height, landmark.z * frame_width]
        for landmark in landmarks
    ])

    row["hand_absent_s"] = 0
    row["landmarks_2d_json"] = json.dumps(np.round(points_2d, 2).tolist())
    row["palm_orientation_2d_deg"] = round(
        float(palm_direction_2d(points_2d)),
        2,
    )

    normal = palm_normal_3d(points_3d)
    if normal is not None:
        raw_axis_angles = np.degrees(np.arccos(np.clip(normal, -1, 1)))
        filtered_axis_angles = []

        for axis, component, raw_angle in zip(AXES, normal, raw_axis_angles):
            filtered_angle = normal_filters[axis].update(float(raw_angle), now)
            row[f"palm_normal_{axis}"] = round(float(component), 5)
            row[f"palm_normal_{axis}_angle_raw_deg"] = round(float(raw_angle), 2)
            row[f"palm_normal_{axis}_angle_filtered_deg"] = round(filtered_angle, 2)
            filtered_axis_angles.append(filtered_angle)

        normal_history.append([frame_number, *filtered_axis_angles])
        cv2.putText(
            frame,
            f"Normal X:{normal[0]:.2f} Y:{normal[1]:.2f} Z:{normal[2]:.2f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    for first, second in HAND_LINES:
        cv2.line(
            frame,
            tuple(points_2d[first].astype(int)),
            tuple(points_2d[second].astype(int)),
            (0, 220, 0),
            2,
        )

    for finger_name, joint_numbers in FINGERS.items():
        first, middle, last = joint_numbers
        raw_angle = joint_bend_angle(points_2d, first, middle, last)
        if raw_angle is None:
            continue

        filtered_angle = finger_filters[finger_name].update(raw_angle, now)
        row[f"{finger_name.lower()}_raw_deg"] = round(raw_angle, 2)
        row[f"{finger_name.lower()}_filtered_deg"] = round(filtered_angle, 2)

        if finger_name == "Index":
            bend_history.append([frame_number, raw_angle, filtered_angle])
            cv2.putText(
                frame,
                f"Index Raw: {raw_angle:.1f} deg",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Index Filtered: {filtered_angle:.1f} deg",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )


if __name__ == "__main__":
    main()
