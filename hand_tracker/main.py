"""Run the hand-tracking application.

The program reads frames from the webcam, asks MediaPipe for 21 hand
landmarks, calculates angles, and optionally sends frames to Depth Anything V2.
"""
print('w')
import json
import sys
import time
import traceback
from collections import deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from hand_tracker.angles import finger_angles, palm_normal_angles, reset_angles
from hand_tracker.data_logging import (
    create_session_paths,
    measurement_fields,
    open_session_csv,
    write_measurement_row,
)
from hand_tracker.depth_estimation import DepthMapWorker
from hand_tracker.measurements import (
    AXES,
    FINGERS,
    bend_angle_world,
    depth_relative_coordinates,
    landmarks_to_pixels,
    palm_normal,
    palm_orientation,
    sample_depth_neighborhood,
    world_landmarks_to_points,
)
from hand_tracker.tracking import create_hand_landmarker, detect_landmarks
from hand_tracker.visualization import (
    draw_depth_status,
    draw_hand_connections,
    draw_index_bend,
    draw_palm_normal,
    draw_performance_overlay,
    draw_trial_status,
    render_live_graphs,
    save_plots,
    show_depth_comparison,
)
from hand_tracker.mujoco_hand import MuJoCoHandViewer


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_FOLDER / "hand_landmarker.task"
OUTPUT_FOLDER = PROJECT_FOLDER / "hand_tracker_data"
DEPTH_SOURCE_FOLDER = PROJECT_FOLDER / "models" / "Depth-Anything-V2"
if not (DEPTH_SOURCE_FOLDER / "depth_anything_v2" / "dpt.py").is_file():
    existing_source_folder = PROJECT_FOLDER / "models" / "Depths_Anything_V2"
    if (existing_source_folder / "depth_anything_v2" / "dpt.py").is_file():
        DEPTH_SOURCE_FOLDER = existing_source_folder
DEPTH_WEIGHTS_FILE = (
    PROJECT_FOLDER / "models" / "weights" / "depth_anything_v2_vits.pth"
)

TRIAL_GAP_SECONDS = 10.0
MAX_HISTORY_POINTS = 100
DEPTH_INPUT_SIZE = 320
DEPTH_SAMPLE_RADIUS = 3
DEPTH_STALE_AFTER_MS = 2500
FPS_WINDOW_SIZE = 60
GRAPH_EVERY_N_FRAMES = 3
MODEL_ANGLE_SOURCE = "filtered"  # "raw" | "filtered"
ANGLE_SMOOTH = 0.5  

# Untuned flip-detection guesses; validate with live collected logs.
FLIP_JUMP_M = 0.04
IMAGE_JUMP_NORM = 0.12
REFLECTION_CLOSE_RATIO = 2.0
FLIP_FIX = "hold"  # "hold" | "mirror" | "off"

class DepthSamples:
    """Hold the newest depth samples and whether each one is valid."""

    def __init__(self):
        """Initialize an empty set of 21 landmark depth samples."""
        self.source_frame = -1
        self.values = np.full(21, np.nan, dtype=float)
        self.validity = np.zeros(21, dtype=bool)

    def update(self, depth_result):
        """Copy depth samples if the worker has a new source frame."""
        if depth_result is None:
            return False
        if depth_result.source_frame == self.source_frame:
            return False

        self.source_frame = depth_result.source_frame
        if depth_result.source_landmarks:
            self.values, self.validity = sample_depth_neighborhood(
                depth_result.depth_map,
                depth_result.source_landmarks,
                radius=DEPTH_SAMPLE_RADIUS,
                return_validity=True,
            )
        else:
            self.values[:] = np.nan
            self.validity[:] = False
        return True


class HandFlipDetector:
    """Detect mirror-like world-pose jumps while ignoring fast image jumps."""

    def __init__(self):
        """Initialize the last image, world pose, and handedness."""
        self.last_image_points = None
        self.last_world_points = None
        self.last_handedness = None

    def update(self, world_points, image_points, handedness):
        """Return pose jump, suspect status, action, and the frame to use."""
        world_points = np.asarray(world_points, dtype=float)
        image_points = np.asarray([
            [landmark.x, landmark.y]
            for landmark in image_points
        ]) if len(image_points) else None
        label = handedness[0] if handedness else ""
        score = handedness[1] if handedness else 0.0

        image_jump = None
        if (image_points is not None
            and self.last_image_points is not None
            and image_points.shape == self.last_image_points.shape
        ):
            image_jump = float(np.mean(np.linalg.norm(
                image_points - self.last_image_points,
                axis=1,
            )))

        image_was_stable = (
            image_jump is None
            or image_jump <= IMAGE_JUMP_NORM
        )
        if image_points is not None:
            self.last_image_points = image_points.copy()

        if self.last_world_points is None:
            world_pose_jump = 0.0
        else:
            world_pose_jump = float(np.mean(np.linalg.norm(
                world_points - self.last_world_points,
                axis=1,
            )))

        handedness_changed = (
            self.last_handedness is not None
            and label
            and label != self.last_handedness
        )
        if label:
            self.last_handedness = label

        closest_reflected = None
        closest_reflected_jump = float("inf")
        for axis in range(3):
            reflected = world_points.copy()
            reflected[:, axis] *= -1.0
            reflected_jump = (
                float("inf")
                if self.last_world_points is None
                else float(np.mean(np.linalg.norm(
                    reflected - self.last_world_points,
                    axis=1,
                )))
            )
            if reflected_jump < closest_reflected_jump:
                closest_reflected_jump = reflected_jump
                closest_reflected = reflected

        mirror_like = (
            self.last_world_points is not None
            and world_pose_jump > FLIP_JUMP_M
            and closest_reflected_jump * REFLECTION_CLOSE_RATIO
            < world_pose_jump
        )
        flip_suspect = bool(
            handedness_changed
            or (image_was_stable and mirror_like)
        )

        action = ""
        accepted_world = world_points
        if flip_suspect:
            if FLIP_FIX == "hold":
                action = "hold"
                accepted_world = None
            elif FLIP_FIX == "mirror" and closest_reflected is not None:
                action = "mirror"
                accepted_world = closest_reflected

        if accepted_world is not None:
            self.last_world_points = accepted_world.copy()

        return {
            "handedness_label": label,
            "handedness_score": score,
            "world_pose_jump_m": world_pose_jump,
            "flip_suspect": int(flip_suspect),
            "flip_action": action,
            "accepted_world": accepted_world,
        }

def log_depth_landmarks(row, depth_result, depth_samples):
    """Write one result for all 21 depth landmarks."""
    if not depth_result.source_landmarks:
        return

    relative_points = depth_relative_coordinates(
        depth_result.source_landmarks,
        depth_samples.values,
    )
    for index, landmark in enumerate(depth_result.source_landmarks):
        prefix = f"landmark_{index:02d}"
        row[f"{prefix}_dav2_source_mp_x_norm"] = round(landmark[0], 6)
        row[f"{prefix}_dav2_source_mp_y_norm"] = round(landmark[1], 6)
        row[f"{prefix}_dav2_source_mp_z_relative"] = round(landmark[2], 6)
        row[f"{prefix}_dav2_depth_valid"] = int(depth_samples.validity[index])

        if not depth_samples.validity[index]:
            continue

        raw_depth = depth_samples.values[index]
        point = relative_points[index]
        depth_value = round(float(raw_depth), 6)
        row[f"{prefix}_dav2_depth_raw_relative"] = depth_value
        row[f"{prefix}_dav2_depth_filtered_relative"] = depth_value
        row[f"{prefix}_dav2_x_norm"] = round(float(point[0]), 6)
        row[f"{prefix}_dav2_y_norm"] = round(float(point[1]), 6)
        row[f"{prefix}_dav2_z_relative"] = round(float(point[2]), 6)

smoothed_angles = {}
def smooth_angle(name, raw_angle):
    """Blend the new angle into the previous smoothed angle."""
    previous = smoothed_angles.get(name)
    if previous is None:
        smoothed_angles[name] = raw_angle
    else:
        smoothed_angles[name] = previous + ANGLE_SMOOTH * (raw_angle - previous)
    return smoothed_angles[name]

def process_hand(
    frame,
    row,
    landmarks,
    world_points,
    frame_number,
    bend_history,
    normal_history,
    mujoco_viewer=None,
):
    """Calculate angles and draw them for a detected hand."""
    points_2d = landmarks_to_pixels(landmarks, frame.shape)
    row["hand_absent_s"] = 0
    row["landmarks_2d_json"] = json.dumps(np.round(points_2d, 2).tolist())
    row["palm_orientation_2d_deg"] = round(
        float(palm_orientation(points_2d)),
        2,
    )

    for index, landmark in enumerate(landmarks):
        prefix = f"landmark_{index:02d}"
        row[f"{prefix}_mp_x_norm"] = round(float(landmark.x), 6)
        row[f"{prefix}_mp_y_norm"] = round(float(landmark.y), 6)
        row[f"{prefix}_mp_z_relative"] = round(float(landmark.z), 6)

    for axis in AXES:
        palm_normal_angles[axis]["raw"] = None
        palm_normal_angles[axis]["filtered"] = None

    normal = palm_normal(world_points)
    if normal is not None:
        raw_axis_angles = np.degrees(np.arccos(np.clip(normal, -1, 1)))

        for axis, component, raw_angle in zip(AXES, normal, raw_axis_angles):
            raw_text = round(float(raw_angle), 2)

            row[f"palm_normal_{axis}"] = round(float(component), 5)
            row[f"palm_normal_{axis}_angle_raw_deg"] = raw_text
            row[f"palm_normal_{axis}_angle_filtered_deg"] = raw_text
            palm_normal_angles[axis]["raw"] = raw_text
            palm_normal_angles[axis]["filtered"] = raw_text

        normal_history.append([frame_number, *raw_axis_angles])
        draw_palm_normal(frame, normal)

    draw_hand_connections(frame, points_2d)
    raw_finger_angles = {}
    filtered_finger_angles = {}
    for finger_name, (first, middle, last) in FINGERS.items():
        current_angle = finger_angles[finger_name.lower()]
        raw_world = bend_angle_world(world_points, first, middle, last)
        raw_finger_angles[finger_name] = raw_world
        filtered_world = smooth_angle(finger_name, raw_world) if raw_world is not None else None
        filtered_finger_angles[finger_name] = filtered_world

        row[f"{finger_name.lower()}_raw_world_deg"] = (
            round(float(raw_world), 2) if raw_world is not None else ""
        )
        raw_angle = raw_world

        if raw_angle is None:
            current_angle["raw"] = None
            current_angle["filtered"] = None
            continue

        raw_text = round(float(raw_angle), 2)
        row[f"{finger_name.lower()}_raw_deg"] = raw_text
        filtered_text = round(float(filtered_world), 2)
        row[f"{finger_name.lower()}_filtered_deg"] = filtered_text
        current_angle["filtered"] = filtered_text
        if finger_name == "Index":
            bend_history.append([frame_number, filtered_world])
            draw_index_bend(frame,filtered_world)

    if mujoco_viewer is not None:
        palm_direction = world_points[9] - world_points[0]
        if MODEL_ANGLE_SOURCE == "filtered":
            model_angles = filtered_finger_angles
        else:
            model_angles = raw_finger_angles
        mujoco_viewer.update(model_angles, normal, palm_direction)  


def open_depth_worker():
    """Try to start the depth worker. Return None if setup is unavailable."""
    try:
        return DepthMapWorker(
            DEPTH_SOURCE_FOLDER,
            DEPTH_WEIGHTS_FILE,
            input_size=DEPTH_INPUT_SIZE,
        )
    except Exception as error:
        print(
            f"[Depth] Unavailable; MediaPipe will continue: {error}",
            flush=True,
        )
        traceback.print_exc()
        return None


def main():
    """Run the webcam loop, calculate hand measurements, and save session output."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")

    session, timestamp, csv_path, plot_folder = create_session_paths(
        OUTPUT_FOLDER
    )
    csv_fields = measurement_fields()
    bend_history = deque(maxlen=MAX_HISTORY_POINTS)
    normal_history = deque(maxlen=MAX_HISTORY_POINTS)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try camera index 1.")

    start_time = time.monotonic()
    last_hand_time = None
    frame_number = 0
    trial_number = 0
    last_timestamp_ms = -1
    depth_samples = DepthSamples()
    camera_frame_times = deque(maxlen=FPS_WINDOW_SIZE)
    mediapipe_times = deque(maxlen=FPS_WINDOW_SIZE)
    live_graph_image = None
    show_depth_view = False
    show_live_graphs = False
    depth_window_open = False
    depth_error = None
    depth_error_reported = False
    # Depth Anything starts OFF. Press D to load it and start computing,
    # press D again to stop it and free the model.
    depth_worker = None
    flip_detector = HandFlipDetector()
    try:
        mujoco_viewer = MuJoCoHandViewer()
        print("[MuJoCo] 3D hand model opened", flush=True)
    except Exception as error:
        mujoco_viewer = None
        print(
            f"[MuJoCo] 3D hand model unavailable: {error}",
            flush=True,
        )

    try:
        with create_hand_landmarker(MODEL_FILE) as tracker:
            with open_session_csv(csv_path, csv_fields) as (writer, csv_file):
                while True:
                    loop_start = time.perf_counter()
                    camera_ok, frame = camera.read()
                    if not camera_ok:
                        break

                    camera_frame_times.append(loop_start)
                    now = time.monotonic()
                    elapsed = now - start_time
                    frame_number += 1
                    timestamp_ms = max(
                        last_timestamp_ms + 1,
                        int(elapsed * 1000),
                    )
                    last_timestamp_ms = timestamp_ms

                    mediapipe_start = time.perf_counter()
                    hands, world_hands, handedness = detect_landmarks(
                        tracker,
                        frame,
                        timestamp_ms,
                    )
                    mediapipe_times.append(time.perf_counter() - mediapipe_start)

                    camera_fps = calculate_fps(camera_frame_times)
                    total_mediapipe_time = sum(mediapipe_times)
                    mediapipe_fps = (
                        len(mediapipe_times) / total_mediapipe_time
                        if total_mediapipe_time > 0
                        else 0.0
                    )
                    hand_detected = bool(hands)
                    landmarks = hands[0] if hand_detected else ()
                    world_landmarks = world_hands[0] if hand_detected else ()
                    detected_world_points = (
                        world_landmarks_to_points(world_landmarks)
                        if hand_detected
                        else None
                    )

                    flip_result = {
                        "handedness_label": handedness[0] if handedness else "",
                        "handedness_score": handedness[1] if handedness else 0.0,
                        "world_pose_jump_m": 0.0,
                        "flip_suspect": 0,
                        "flip_action": "",
                        "accepted_world": detected_world_points,
                    }
                    if detected_world_points is not None:
                        flip_result = flip_detector.update(
                            detected_world_points,
                            landmarks,
                            handedness,
                        )
                    accepted_world = flip_result["accepted_world"]
                    accepted_hand = (
                        hand_detected
                        and accepted_world is not None
                    )

                    starts_new_trial = accepted_hand and (
                        last_hand_time is None
                        or now - last_hand_time > TRIAL_GAP_SECONDS
                    )
                    if starts_new_trial:
                        trial_number += 1

                    if depth_worker is not None:
                        depth_worker.submit(
                            frame,
                            frame_number,
                            now,
                            landmarks,
                        )
                        depth_result = depth_worker.latest()
                        worker_error = depth_worker.error
                        if worker_error is not None and not depth_error_reported:
                            depth_error = str(worker_error)
                            depth_error_reported = True
                            print(
                                "[Depth] Worker unavailable; MediaPipe "
                                f"will continue: {worker_error}",
                                flush=True,
                            )
                        depth_performance = depth_worker.performance()
                    else:
                        depth_result = None
                        depth_performance = {
                            "inference_fps": 0.0,
                            "average_inference_ms": 0.0,
                            "input_size": DEPTH_INPUT_SIZE,
                        }

                    new_depth_result = depth_samples.update(depth_result)
                    row = {field: "" for field in csv_fields}
                    row.update({
                        "timestamp": time.time(),
                        "elapsed_s": round(elapsed, 4),
                        "frame": frame_number,
                        "source_frame_number": frame_number,
                        "trial": trial_number,
                        "hand_detected": int(hand_detected),
                        "camera_fps": round(camera_fps, 3),
                        "mediapipe_fps": round(mediapipe_fps, 3),
                        "depth_input_size": DEPTH_INPUT_SIZE,
                        "handedness_label":
                            flip_result["handedness_label"],
                        "handedness_score":
                            round(float(flip_result["handedness_score"]), 4),
                        "world_pose_jump_m":
                            round(float(flip_result["world_pose_jump_m"]), 6),
                        "flip_suspect":
                            flip_result["flip_suspect"],
                        "flip_action":
                            flip_result["flip_action"],
                    })

                    if depth_result is not None:
                        age_ms = max(
                            (now - depth_result.source_timestamp) * 1000,
                            0,
                        )
                        row.update({
                            "dav2_source_frame": depth_result.source_frame,
                            "dav2_age_frames":
                                frame_number - depth_result.source_frame,
                            "dav2_age_ms": round(age_ms, 2),
                            "dav2_inference_ms":
                                round(depth_result.inference_ms, 2),
                            "dav2_inference_fps":
                                round(depth_result.inference_fps, 3),
                            "depth_inference_frame_number":
                                depth_result.source_frame,
                            "depth_result_age_ms": round(age_ms, 2),
                            "dav2_average_inference_ms":
                                round(depth_performance["average_inference_ms"], 2),
                            "dav2_completion_fps":
                                round(depth_performance["inference_fps"], 3),
                        })

                    if new_depth_result and depth_result is not None:
                        log_depth_landmarks(
                            row,
                            depth_result,
                            depth_samples,
                        )

                    if starts_new_trial:
                        print(f"Trial {trial_number} started")

                    if accepted_hand:
                        process_hand(
                            frame,
                            row,
                            landmarks,
                            accepted_world,
                            frame_number,
                            bend_history,
                            normal_history,
                            mujoco_viewer,
                        )
                        last_hand_time = now
                    else:
                        reset_angles()
                        if last_hand_time is not None:
                            absent_seconds = now - last_hand_time
                            row["hand_absent_s"] = round(absent_seconds, 3)
                            if absent_seconds > TRIAL_GAP_SECONDS:
                                last_hand_time = None

                    write_measurement_row(writer, csv_file, row)
                    draw_trial_status(frame, trial_number, frame_number)

                    worker_alive = (
                        depth_worker is not None and depth_worker.is_alive
                    )
                    depth_age_ms = (
                        max(
                            (now - depth_result.source_timestamp) * 1000,
                            0,
                        )
                        if depth_result is not None
                        else None
                    )
                    depth_is_stale = (
                        depth_age_ms is not None
                        and depth_age_ms > DEPTH_STALE_AFTER_MS
                    )
                    draw_depth_status(
                        frame,
                        show_depth_view,
                        worker_alive,
                        depth_result,
                        now,
                        error=depth_error,
                        stale_after_ms=DEPTH_STALE_AFTER_MS,
                    )
                    draw_performance_overlay(
                        frame,
                        camera_fps,
                        mediapipe_fps,
                        depth_performance["inference_fps"],
                        depth_performance["average_inference_ms"],
                        depth_age_ms,
                        depth_performance["input_size"],
                        stale=depth_is_stale,
                    )
                    cv2.imshow("Hand Tracker", frame)
                    if mujoco_viewer is not None:
                        mujoco_image = mujoco_viewer.render()
                        cv2.imshow("MuJoCo 3D Hand", mujoco_image)
                        mujoco_viewer.set_mouse_callback()
                    if show_live_graphs:
                        if (
                            live_graph_image is None
                            or frame_number % GRAPH_EVERY_N_FRAMES == 0
                        ):
                            live_graph_image = render_live_graphs(
                                bend_history,
                                normal_history,
                            )
                        cv2.imshow("Live Graphs", live_graph_image)

                    if (
                        show_depth_view
                        and depth_result is not None
                        and (new_depth_result or not depth_window_open)
                    ):
                        show_depth_comparison_window(
                            depth_result,
                            frame_number,
                            depth_samples,
                            depth_performance,
                            depth_is_stale,
                        )
                        depth_window_open = True

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if mujoco_viewer is not None and mujoco_viewer.handle_key(key):
                        continue
                    if key == ord("r"):
                        reset_angles()
                        print("Angle values cleared.")
                    elif key == ord("p"):
                        save_plots(
                            bend_history,
                            normal_history,
                            plot_folder,
                            session,
                            timestamp,
                        )
                        print(f"Plots saved in {plot_folder}")
                    elif key == ord("d"):
                        if depth_worker is None:
                            # Turn depth ON: load the model and start computing.
                            # This can pause the video for a moment while the model loads.
                            print("[Depth] Starting...", flush=True)
                            depth_worker = open_depth_worker()
                            depth_error = None
                            depth_error_reported = False
                            show_depth_view = depth_worker is not None
                            print(
                                "[Depth] "
                                f"{'ON' if depth_worker is not None else 'could not start'}",
                                flush=True,
                            )
                        else:
                            # Turn depth OFF: stop the worker and clear its data.
                            depth_worker.close()
                            depth_worker = None
                            depth_samples = DepthSamples()
                            show_depth_view = False
                            print("[Depth] OFF", flush=True)
                            if depth_window_open:
                                cv2.destroyWindow("Depth Anything V2 comparison")
                                depth_window_open = False
                    elif key == ord("g"):
                        show_live_graphs = not show_live_graphs
                        live_graph_image = None
                        print(
                            "[Graphs] Live graphs "
                            f"{'ON' if show_live_graphs else 'OFF'}",
                            flush=True,
                        )
                        if show_live_graphs:
                            live_graph_image = render_live_graphs(
                                bend_history,
                                normal_history,
                            )
                            cv2.imshow(
                                "Live Graphs",
                                live_graph_image,
                            )
                        else:
                            cv2.destroyWindow("Live Graphs")
    finally:
        if depth_worker is not None:
            depth_worker.close()
        if mujoco_viewer is not None:
            mujoco_viewer.close()
        camera.release()
        cv2.destroyAllWindows()

    save_plots(
        bend_history,
        normal_history,
        plot_folder,
        session,
        timestamp,
    )
    print(f"CSV saved: {csv_path}")
    print(f"Plots saved in: {plot_folder}")


def calculate_fps(times):
    """Calculate FPS from evenly timed recent loop or processing samples."""
    if len(times) < 2:
        return 0.0
    duration = times[-1] - times[0]
    if duration <= 0:
        return 0.0
    return (len(times) - 1) / duration


def show_depth_comparison_window(
    depth_result,
    frame_number,
    depth_samples,
    depth_performance,
    depth_is_stale,
):
    """Display the source RGB frame beside its relative-depth map."""
    if depth_result.source_landmarks:
        points = np.array([
            [
                x * (depth_result.depth_map.shape[1] - 1),
                y * (depth_result.depth_map.shape[0] - 1),
            ]
            for x, y, _ in depth_result.source_landmarks
        ])
    else:
        points = None

    show_depth_comparison(
        depth_result.source_image,
        depth_result.depth_map,
        points,
        depth_samples.values if depth_result.source_landmarks else None,
        source_frame=depth_result.source_frame,
        age_frames=frame_number - depth_result.source_frame,
        inference_ms=depth_result.inference_ms,
        inference_fps=depth_result.inference_fps,
        average_inference_ms=depth_performance["average_inference_ms"],
        input_size=depth_performance["input_size"],
        landmark_validity=(
            depth_samples.validity
            if depth_result.source_landmarks
            else None
        ),
        stale=depth_is_stale,
    )


if __name__ == "__main__":
    main()
