import csv
import cv2
import math
import os
import re
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from filterpy.kalman import KalmanFilter

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
HISTORY_LEN = 200
INDEX_MJB_IDS = (5, 6, 7)  # MCP, PIP, DIP -> angle at PIP

CAM_LEFT_INDEX = 0
CAM_RIGHT_INDEX = 1

# --- Stereo rig parameters: MUST be set for your physical setup ---
# FOCAL_PX: focal length in pixels. Rough estimate = (frame_width_px / 2) / tan(HFOV/2).
# BASELINE_CM: physical distance between the two camera lenses, in cm.
# Getting these exactly right matters for absolute depth values; angle
# calculations are far less sensitive to a uniform scale error in both,
# since angle_between() normalizes vectors before computing angles.
FOCAL_PX = 1050
BASELINE_CM = 10.16

os.makedirs(PLOT_DIR, exist_ok=True)


class AngleFilter:
    """Kalman filter with outlier rejection, streak-based re-anchoring,
    velocity clamping, and output clamping. Prevents the filter from
    running away when measurements are rejected for too many frames
    in a row (the bug found and fixed in the monocular pipeline)."""

    def __init__(self, max_jump=30.0, max_consecutive_rejects=6, max_velocity=60.0,
                 output_min=0.0, output_max=180.0):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.0], [0.0]])
        self.kf.F = np.array([[1, 1], [0, 1]])
        self.kf.H = np.array([[1, 0]])
        self.initial_P = np.eye(2) * 1000.0
        self.kf.P = self.initial_P.copy()
        self.kf.R = 5
        self.kf.Q = np.array([[0.01, 0], [0, 0.01]])
        self.started = False
        self.max_jump = max_jump
        self.max_consecutive_rejects = max_consecutive_rejects
        self.max_velocity = max_velocity
        self.output_min = output_min
        self.output_max = output_max
        self.reject_streak = 0

    def _clamp_velocity(self):
        self.kf.x[1, 0] = float(np.clip(self.kf.x[1, 0], -self.max_velocity, self.max_velocity))

    def _clamped_output(self):
        return float(np.clip(self.kf.x[0, 0], self.output_min, self.output_max))

    def update(self, value):
        if not self.started:
            self.kf.x[0, 0] = value
            self.started = True
            self.kf.predict()
            self.kf.update(value)
            self._clamp_velocity()
            return self._clamped_output()

        last_state = self.kf.x[0, 0]
        self.kf.predict()

        if abs(value - last_state) > self.max_jump:
            self.reject_streak += 1
            if self.reject_streak >= self.max_consecutive_rejects:
                print(f"Filter re-anchored after {self.reject_streak} rejects: {value:.1f}")
                self.kf.x[0, 0] = value
                self.kf.x[1, 0] = 0.0
                self.kf.P = self.initial_P.copy()
                self.reject_streak = 0
                return self._clamped_output()

            print(f"Outlier rejected: {value:.1f} vs last state {last_state:.1f}")
            return self._clamped_output()

        self.reject_streak = 0
        self.kf.update(value)
        self._clamp_velocity()
        return self._clamped_output()


def create_landmarker():
    opts = base_options.BaseOptions(model_asset_path=MODEL_PATH)
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=opts,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def next_timestamp_ms(elapsed_s, last_timestamp_ms):
    """Convert elapsed seconds to an integer ms timestamp, forcing it
    strictly forward so detect_for_video never sees a repeated or
    backwards value for a given landmarker instance."""
    ts = int(elapsed_s * 1000)
    if ts <= last_timestamp_ms:
        ts = last_timestamp_ms + 1
    return ts


def build_projection_matrices(focal_px, baseline_cm, w, h):
    K = np.array([
        [focal_px, 0, w / 2.0],
        [0, focal_px, h / 2.0],
        [0, 0, 1.0],
    ])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((np.eye(3), np.array([[-baseline_cm], [0], [0]])))
    return P1, P2


def triangulate_landmarks(landmarks_left, landmarks_right, P1, P2, w, h):
    pts_left = np.array([[p.x * w, p.y * h] for p in landmarks_left], dtype=np.float64).T
    pts_right = np.array([[p.x * w, p.y * h] for p in landmarks_right], dtype=np.float64).T
    points_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d  # shape (21, 3), in cm, in the left camera's frame


def angle_between(a, b):
    m = np.linalg.norm(a) * np.linalg.norm(b)
    if m == 0:
        return None
    return math.degrees(math.acos(max(-1, min(1, np.dot(a, b) / m))))


def joint_angle_3d(points_3d, a, b, c):
    return angle_between(points_3d[a] - points_3d[b], points_3d[c] - points_3d[b])


def render_plot(fig, canvas):
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, 1:]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_next_experiment_number():
    pattern = re.compile(r"stereo_angle_(\d{3})\.(?:png|csv)$")
    existing_numbers = []
    if os.path.isdir(PLOT_DIR):
        for file_name in os.listdir(PLOT_DIR):
            match = pattern.match(file_name)
            if match:
                existing_numbers.append(int(match.group(1)))
    return max(existing_numbers, default=0) + 1


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        csv_writer.writerows(rows)
    print(f"Saved CSV to: {path}")


def save_graph(history, experiment_number):
    base_name = f"stereo_angle_{experiment_number:03d}"
    csv_path = os.path.join(PLOT_DIR, f"{base_name}.csv")
    columns = ["Frame", "Angle Raw", "Angle Filtered"]
    rows = list(zip(history["frame"], history["angle_raw"], history["angle_filt"]))
    write_csv(csv_path, columns, rows)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(history["frame"], history["angle_raw"], label="Stereo Raw")
    ax.plot(history["frame"], history["angle_filt"], label="Stereo Filtered")
    ax.set_title("Index MJB Angle (Stereo Triangulated)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True)
    ax.legend()
    image_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    fig.savefig(image_path)
    plt.close(fig)
    print(f"Saved graph to: {image_path}")


def setup_plot():
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set(title="Index MJB Angle (Stereo Triangulated)", xlabel="Frame", ylabel="Angle (deg)", ylim=(0, 180))
    ax.grid(True)
    lines = {
        "angle_raw": ax.plot([], [], label="Stereo Raw")[0],
        "angle_filt": ax.plot([], [], label="Stereo Filtered")[0],
    }
    ax.legend()
    return fig, FigureCanvas(fig), ax, lines


def main():
    tracker_left = create_landmarker()
    tracker_right = create_landmarker()
    print("Hand trackers loaded (left + right)")

    fig, canvas, ax, lines = setup_plot()

    cap_left = cv2.VideoCapture(CAM_LEFT_INDEX)
    cap_right = cv2.VideoCapture(CAM_RIGHT_INDEX)
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise SystemExit("Could not open both cameras")

    w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise SystemExit(
            f"Camera returned invalid dimensions ({w}x{h}). "
            "Try setting CAP_PROP_FRAME_WIDTH/HEIGHT explicitly before "
            "reading them back, or check the camera driver."
        )
    P1, P2 = build_projection_matrices(FOCAL_PX, BASELINE_CM, w, h)
    print(f"Cameras opened ({w}x{h}). Using FOCAL_PX={FOCAL_PX}, BASELINE_CM={BASELINE_CM}")

    cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Live Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Plot", 800, 500)

    history = {k: deque(maxlen=HISTORY_LEN) for k in ["frame", "angle_raw", "angle_filt"]}
    angle_filter = AngleFilter()

    experiment_number = get_next_experiment_number()
    print(f"Using experiment number: {experiment_number:03d}")
    start_time = time.perf_counter()
    last_ts_left = -1
    last_ts_right = -1

    try:
        while True:
            # Grab both frames as close together as possible before decoding.
            # This narrows but does not eliminate the sync gap -- true hardware
            # synchronization would require a hardware trigger/genlock.
            # Note: cv2.waitKey() must run every iteration regardless of grab/retrieve
            # success, or the OS will mark the window as unresponsive.
            grabbed_left = cap_left.grab()
            grabbed_right = cap_right.grab()
            if not grabbed_left or not grabbed_right:
                print(f"grab failed -- left: {grabbed_left}, right: {grabbed_right}")
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            success_left, frame_left = cap_left.retrieve()
            success_right, frame_right = cap_right.retrieve()
            if not success_left or not success_right:
                print(f"retrieve failed -- left: {success_left}, right: {success_right}")
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            rgb_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            rgb_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            mpimg_left = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_left)
            mpimg_right = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_right)

            elapsed = time.perf_counter() - start_time
            ts_left = next_timestamp_ms(elapsed, last_ts_left)
            ts_right = next_timestamp_ms(elapsed, last_ts_right)
            last_ts_left, last_ts_right = ts_left, ts_right

            result_left = tracker_left.detect_for_video(mpimg_left, ts_left)
            result_right = tracker_right.detect_for_video(mpimg_right, ts_right)

            if result_left.hand_landmarks and result_right.hand_landmarks:
                points_left = result_left.hand_landmarks[0]
                points_right = result_right.hand_landmarks[0]

                points_3d = triangulate_landmarks(points_left, points_right, P1, P2, w, h)
                angle_raw = joint_angle_3d(points_3d, *INDEX_MJB_IDS)

                if angle_raw is not None:
                    angle_filt = angle_filter.update(angle_raw)

                    idx = history["frame"][-1] + 1 if history["frame"] else 1
                    history["frame"].append(idx)
                    history["angle_raw"].append(angle_raw)
                    history["angle_filt"].append(angle_filt)

                    lines["angle_raw"].set_data(history["frame"], history["angle_raw"])
                    lines["angle_filt"].set_data(history["frame"], history["angle_filt"])
                    ax.set_xlim(max(0, idx - HISTORY_LEN + 1), idx + 1)
                    ax.relim()
                    ax.autoscale_view()

                vision.drawing_utils.draw_landmarks(frame_left, points_left, vision.HandLandmarksConnections.HAND_CONNECTIONS)

            cv2.imshow("Left Camera", frame_left)
            cv2.imshow("Live Plot", render_plot(fig, canvas))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        save_graph(history, experiment_number)


if __name__ == "__main__":
    main()