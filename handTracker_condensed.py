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
PLOT_FINGER = "Index"
FRAME_DIMS = {"w": None, "h": None}
FINGERS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

os.makedirs(PLOT_DIR, exist_ok=True)

class AngleFilter:
    def __init__(self, max_jump=30.0, max_consecutive_rejects=5, max_velocity=60.0):
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
        self.reject_streak = 0

    def update(self, value):
        if not self.started:
            self.kf.x[0, 0] = value
            self.started = True
            self.kf.predict()
            self.kf.update(value)
            self.kf.x[1, 0] = float(np.clip(self.kf.x[1, 0], -self.max_velocity, self.max_velocity))
            return float(np.clip(self.kf.x[0, 0], 0.0, 180.0))

        last_state = self.kf.x[0, 0]
        self.kf.predict()

        if abs(value - last_state) > self.max_jump:
            self.reject_streak += 1
            print(f"Outlier rejected: {value:.1f} vs last state {last_state:.1f}")

            if self.reject_streak >= self.max_consecutive_rejects:
                self.kf.x[0, 0] = last_state
                self.kf.x[1, 0] = 0.0
                self.kf.P = self.initial_P.copy()
                self.kf.x[1, 0] = float(np.clip(self.kf.x[1, 0], -self.max_velocity, self.max_velocity))
                self.reject_streak = 0

            return float(np.clip(self.kf.x[0, 0], 0.0, 180.0))

        self.reject_streak = 0
        self.kf.update(value)
        self.kf.x[1, 0] = float(np.clip(self.kf.x[1, 0], -self.max_velocity, self.max_velocity))
        return float(np.clip(self.kf.x[0, 0], 0.0, 180.0))


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


def coords(p):
    arr = np.asarray(p) if isinstance(p, np.ndarray) else np.array([p.x, p.y, p.z])
    w, h = FRAME_DIMS["w"], FRAME_DIMS["h"]
    if w is None:
        return arr
    return np.array([arr[0] * w, arr[1] * h, arr[2] * w])


def angle_between(a, b):
    m = np.linalg.norm(a) * np.linalg.norm(b)
    if m == 0:
        return None
    return math.degrees(math.acos(max(-1, min(1, np.dot(a, b) / m))))


def joint_angle(points, a, b, c):
    return angle_between(coords(points[a]) - coords(points[b]), coords(points[c]) - coords(points[b]))

def calculate_hand_orientation(points):
    wrist, idx, pinky, middle = (coords(points[i]) for i in (0, 5, 17, 9))
    hand_x = idx - pinky
    hand_y = middle - wrist
    hand_x /= np.linalg.norm(hand_x)
    hand_y /= np.linalg.norm(hand_y)
    palm_normal = np.cross(hand_x, hand_y)
    palm_normal /= np.linalg.norm(palm_normal)
    return hand_x, hand_y, palm_normal


def finger_direction(points, finger_name):
    ids = FINGERS[finger_name]
    mcp = coords(points[ids[0]])
    tip = coords(points[ids[-1]])
    direction = tip - mcp
    norm = np.linalg.norm(direction)
    return direction / norm if norm != 0 else direction


def finger_elevation(points, finger_name):
    _, _, palm_normal = calculate_hand_orientation(points)
    finger_dir = finger_direction(points, finger_name)
    return angle_between(finger_dir, palm_normal)


def render_plot(fig, canvas):
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, 1:]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_next_experiment_number():
    pattern = re.compile(r"graph\d+_(\d{3})\.(?:png|csv)$")
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


def save_graph1(history, experiment_number):
    base_name = f"graph1_{experiment_number:03d}"
    csv_path = os.path.join(PLOT_DIR, f"{base_name}.csv")
    columns = ["Frame", "Camera Raw", "Camera Filtered", "Hand Raw", "Hand Filtered"]
    rows = list(zip(history["frame"], history["cam_raw"], history["cam_filt"], history["hand_raw"], history["hand_filt"]))
    write_csv(csv_path, columns, rows)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(history["frame"], history["cam_raw"], label="Camera Raw")
    ax.plot(history["frame"], history["cam_filt"], label="Camera Filtered")
    ax.plot(history["frame"], history["hand_raw"], label="Hand Raw")
    ax.plot(history["frame"], history["hand_filt"], label="Hand Filtered")
    ax.set_title("Hand-Oriented Index Finger Direction")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True)
    ax.legend()
    image_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    fig.savefig(image_path)
    plt.close(fig)
    print(f"Saved graph to: {image_path}")


def save_graph2(history, experiment_number):
    base_name = f"graph2_{experiment_number:03d}"
    csv_path = os.path.join(PLOT_DIR, f"{base_name}.csv")
    columns = ["Frame", "Palm X", "Palm Y", "Palm Z"]
    rows = list(zip(history["frame"], history["palm_x"], history["palm_y"], history["palm_z"]))
    write_csv(csv_path, columns, rows)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(history["frame"], history["palm_x"], label="Palm X")
    ax.plot(history["frame"], history["palm_y"], label="Palm Y")
    ax.plot(history["frame"], history["palm_z"], label="Palm Z")
    ax.set_title("Filtered Palm Normal Components")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Component")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    ax.legend()
    image_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    fig.savefig(image_path)
    plt.close(fig)
    print(f"Saved graph to: {image_path}")


def setup_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    ax1.set(title="Hand-Oriented Index Finger Direction", xlabel="Frame", ylabel="Angle (deg)", ylim=(0, 180))
    ax2.set(title="Filtered Palm Normal Components", xlabel="Frame", ylabel="Component", ylim=(-1, 1))
    for ax in (ax1, ax2):
        ax.grid(True)
    lines = {
        "cam_raw": ax1.plot([], [], label="Camera MJB")[0],
        "cam_filt": ax1.plot([], [], label="Camera MJB filtered")[0],
        "hand_raw": ax1.plot([], [], label="Hand-oriented raw")[0],
        "hand_filt": ax1.plot([], [], label="Hand-oriented filtered")[0],
        "palm_x": ax2.plot([], [], label="Filtered Palm Normal X")[0],
        "palm_y": ax2.plot([], [], label="Filtered Palm Normal Y")[0],
        "palm_z": ax2.plot([], [], label="Filtered Palm Normal Z")[0],
    }
    ax1.legend()
    ax2.legend()
    return fig, FigureCanvas(fig), ax1, ax2, lines


def main():
    tracker = create_landmarker()
    print("Hand tracker loaded")
    fig, canvas, ax1, ax2, lines = setup_plot()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Camera failed")
    FRAME_DIMS["w"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    FRAME_DIMS["h"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera opened")
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Live Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Plot", 800, 600)

    history = {k: deque(maxlen=HISTORY_LEN) for k in ["frame", "cam_raw", "cam_filt", "hand_raw", "hand_filt", "palm_x", "palm_y", "palm_z"]}
    filters = {f"{finger}_MJB": AngleFilter() for finger in FINGERS}
    hand_orientation_filter = AngleFilter()
    palm_filters = {axis: AngleFilter() for axis in ("palm_x", "palm_y", "palm_z")}
    experiment_number = get_next_experiment_number()
    print(f"Using experiment number: {experiment_number:03d}")
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpimg = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = tracker.detect_for_video(mpimg, timestamp_ms)

            if result.hand_landmarks:
                points = result.hand_landmarks[0]
                hand_x, hand_y, palm_normal = calculate_hand_orientation(points)
                finger_raw = finger_elevation(points, PLOT_FINGER)
                finger_filt = hand_orientation_filter.update(finger_raw)
                raw_palm_x, raw_palm_y, raw_palm_z = palm_normal
                palm_x = palm_filters["palm_x"].update(raw_palm_x)
                palm_y = palm_filters["palm_y"].update(raw_palm_y)
                palm_z = palm_filters["palm_z"].update(raw_palm_z)
                print("Hand-oriented index finger elevation:")
                print(f"Raw: {finger_raw:.3f}")
                print(f"Filtered: {finger_filt:.3f}")
                camera = {
                    name: {
                        "MJB_raw": joint_angle(points, *ids[:3]),
                        "MJB_filtered": filters[f"{name}_MJB"].update(joint_angle(points, *ids[:3])) if joint_angle(points, *ids[:3]) is not None else None,
                    }
                    for name, ids in FINGERS.items()
                }
                if camera[PLOT_FINGER]["MJB_raw"] is not None:
                    idx = history["frame"][-1] + 1 if history["frame"] else 1
                    history["frame"].append(idx)
                    history["cam_raw"].append(camera[PLOT_FINGER]["MJB_raw"])
                    history["cam_filt"].append(camera[PLOT_FINGER]["MJB_filtered"])
                    history["hand_raw"].append(finger_raw)
                    history["hand_filt"].append(finger_filt)
                    history["palm_x"].append(palm_x)
                    history["palm_y"].append(palm_y)
                    history["palm_z"].append(palm_z)

                    lines["cam_raw"].set_data(history["frame"], history["cam_raw"])
                    lines["cam_filt"].set_data(history["frame"], history["cam_filt"])
                    lines["hand_raw"].set_data(history["frame"], history["hand_raw"])
                    lines["hand_filt"].set_data(history["frame"], history["hand_filt"])
                    ax1.set_xlim(max(0, idx - HISTORY_LEN + 1), idx + 1)
                    ax1.relim(); ax1.autoscale_view()

                    lines["palm_x"].set_data(history["frame"], history["palm_x"])
                    lines["palm_y"].set_data(history["frame"], history["palm_y"])
                    lines["palm_z"].set_data(history["frame"], history["palm_z"])
                    ax2.set_xlim(max(0, idx - HISTORY_LEN + 1), idx + 1)
                    ax2.relim(); ax2.autoscale_view()

                vision.drawing_utils.draw_landmarks(frame, points, vision.HandLandmarksConnections.HAND_CONNECTIONS)

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Live Plot", render_plot(fig, canvas))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_graph1(history, experiment_number)
        save_graph2(history, experiment_number)


if __name__ == "__main__":
    main()