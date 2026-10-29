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
FRAME_DIMS = {"w": None, "h": None}

os.makedirs(PLOT_DIR, exist_ok=True)


class AngleFilter:
    def __init__(self, max_jump=100.0, max_consecutive_rejects=5):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.0], [0.0]])
        self.kf.F = np.array([[1, 1], [0, 1]])
        self.kf.H = np.array([[1, 0]])
        self.kf.P *= 200
        self.kf.R = 5
        self.kf.Q = np.array([[0.01, 0], [0, 0.01]])
        self.started = False
        self.max_jump = max_jump
        self.max_consecutive_rejects = max_consecutive_rejects
        self.reject_streak = 0

    def update(self, value):
        if not self.started:
            self.kf.x[0, 0] = value
            self.started = True
            self.kf.predict()
            self.kf.update(value)
            return float(self.kf.x[0, 0])

        last_state = self.kf.x[0, 0]
        self.kf.predict()

        if abs(value - last_state) > self.max_jump:
            self.reject_streak += 1
            if self.reject_streak >= self.max_consecutive_rejects:
                # We've rejected too many in a row -- the filter has lost
                # lock, not the measurements. Re-anchor to the incoming
                # value and reset velocity so it stops extrapolating.
                print(f"Filter re-anchored after {self.reject_streak} rejects: {value:.1f}")
                self.kf.x[0, 0] = value
                self.kf.x[1, 0] = 0.0
                self.kf.P *= 1000
                self.reject_streak = 0
                return float(self.kf.x[0, 0])

            print(f"Outlier rejected: {value:.1f} vs last state {last_state:.1f}")
            return float(self.kf.x[0, 0])

        self.reject_streak = 0
        self.kf.update(value)
        return float(self.kf.x[0, 0])

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
    arr = np.array([p.x, p.y, p.z])
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


def render_plot(fig, canvas):
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, 1:]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_next_experiment_number():
    pattern = re.compile(r"deriv_(\d{3})\.(?:png|csv)$")
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
    base_name = f"deriv_{experiment_number:03d}"
    csv_path = os.path.join(PLOT_DIR, f"{base_name}.csv")
    columns = ["Frame", "Angle Raw", "Angle Filtered", "Raw Derivative (deg/s)", "Filtered Derivative (deg/s)"]
    rows = list(zip(
        history["frame"], history["angle_raw"], history["angle_filt"],
        history["deriv_raw"], history["deriv_filt"],
    ))
    write_csv(csv_path, columns, rows)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(history["frame"], history["deriv_raw"], label="Raw Derivative")
    ax.plot(history["frame"], history["deriv_filt"], label="Filtered Derivative")
    ax.set_title("Index MJB Angular Velocity (deg/s)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angular velocity (deg/s)")
    ax.grid(True)
    ax.legend()
    image_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    fig.savefig(image_path)
    plt.close(fig)
    print(f"Saved graph to: {image_path}")


def setup_plot():
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set(title="Index MJB Angular Velocity (deg/s)", xlabel="Frame", ylabel="Angular velocity (deg/s)")
    ax.grid(True)
    lines = {
        "deriv_raw": ax.plot([], [], label="Raw Derivative")[0],
        "deriv_filt": ax.plot([], [], label="Filtered Derivative")[0],
    }
    ax.legend()
    return fig, FigureCanvas(fig), ax, lines


def main():
    tracker = create_landmarker()
    print("Hand tracker loaded")
    fig, canvas, ax, lines = setup_plot()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Camera failed")
    FRAME_DIMS["w"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    FRAME_DIMS["h"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera opened")
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Live Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Plot", 800, 500)

    history = {k: deque(maxlen=HISTORY_LEN) for k in
               ["frame", "angle_raw", "angle_filt", "deriv_raw", "deriv_filt"]}
    angle_filter = AngleFilter()

    experiment_number = get_next_experiment_number()
    print(f"Using experiment number: {experiment_number:03d}")
    start_time = time.perf_counter()

    prev_raw = None
    prev_filt = None
    prev_t = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpimg = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
            now = time.perf_counter() - start_time
            timestamp_ms = int(now * 1000)
            result = tracker.detect_for_video(mpimg, timestamp_ms)

            if result.hand_landmarks:
                points = result.hand_landmarks[0]
                angle_raw = joint_angle(points, *INDEX_MJB_IDS)

                if angle_raw is not None:
                    angle_filt = angle_filter.update(angle_raw)

                    # Derivative = rate of change per second, using actual
                    # elapsed time between successful detections (not just
                    # frame count), so gaps from missed detections don't
                    # distort the rate.
                    if prev_t is not None and now > prev_t:
                        dt = now - prev_t
                        deriv_raw = (angle_raw - prev_raw) / dt
                        deriv_filt = (angle_filt - prev_filt) / dt
                    else:
                        deriv_raw = 0.0
                        deriv_filt = 0.0

                    prev_raw, prev_filt, prev_t = angle_raw, angle_filt, now

                    idx = history["frame"][-1] + 1 if history["frame"] else 1
                    history["frame"].append(idx)
                    history["angle_raw"].append(angle_raw)
                    history["angle_filt"].append(angle_filt)
                    history["deriv_raw"].append(deriv_raw)
                    history["deriv_filt"].append(deriv_filt)

                    lines["deriv_raw"].set_data(history["frame"], history["deriv_raw"])
                    lines["deriv_filt"].set_data(history["frame"], history["deriv_filt"])
                    ax.set_xlim(max(0, idx - HISTORY_LEN + 1), idx + 1)
                    ax.relim(); ax.autoscale_view()

                vision.drawing_utils.draw_landmarks(frame, points, vision.HandLandmarksConnections.HAND_CONNECTIONS)

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Live Plot", render_plot(fig, canvas))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_graph(history, experiment_number)


if __name__ == "__main__":
    main()