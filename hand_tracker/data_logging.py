"""Create CSV session names and write measurement rows."""

import csv
import time
from contextlib import contextmanager

from .measurements import AXES, FINGERS

LANDMARK_COUNT = 21


def create_session_paths(output_dir):
    """Make the output folders and choose the next session number."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sessions = [
        int(path.stem.split("_")[1])
        for path in output_dir.glob("session_*.csv")
        if len(path.stem.split("_")) > 1
        and path.stem.split("_")[1].isdigit()
    ]
    session = max(sessions, default=0) + 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"session_{session:03d}_{stamp}.csv"
    return session, stamp, csv_path, plot_dir


def measurement_fields():
    """Build the ordered CSV header, preserving all existing columns."""
    fields = [
        "timestamp", "elapsed_s", "frame", "trial", "hand_detected",
        "hand_absent_s", "palm_orientation_2d_deg", "landmarks_2d_json",
    ]
    for axis in AXES:
        fields += [
            f"palm_normal_{axis}",
            f"palm_normal_{axis}_angle_raw_deg",
            f"palm_normal_{axis}_angle_filtered_deg",
        ]
    for name in FINGERS:
        fields += [f"{name.lower()}_raw_deg", f"{name.lower()}_filtered_deg"]
    fields += [
        "dav2_source_frame", "dav2_age_frames", "dav2_age_ms",
        "dav2_inference_ms", "dav2_inference_fps",
    ]
    for index in range(LANDMARK_COUNT):
        prefix = f"landmark_{index:02d}"
        fields += [
            f"{prefix}_mp_x_norm",
            f"{prefix}_mp_y_norm",
            f"{prefix}_mp_z_relative",
            f"{prefix}_dav2_source_mp_x_norm",
            f"{prefix}_dav2_source_mp_y_norm",
            f"{prefix}_dav2_source_mp_z_relative",
            f"{prefix}_dav2_depth_raw_relative",
            f"{prefix}_dav2_depth_filtered_relative",
            f"{prefix}_dav2_x_norm",
            f"{prefix}_dav2_y_norm",
            f"{prefix}_dav2_z_relative",
            f"{prefix}_stereo_depth_m",
        ]
    fields += [
        "source_frame_number",
        "depth_inference_frame_number",
        "depth_result_age_ms",
        "depth_input_size",
        "camera_fps",
        "mediapipe_fps",
        "dav2_average_inference_ms",
        "dav2_completion_fps",
    ]
    for index in range(LANDMARK_COUNT):
        fields.append(f"landmark_{index:02d}_dav2_depth_valid")

    # New 3D raw-angle columns are appended after every existing CSV column.
    for name in FINGERS:
        fields.append(f"{name.lower()}_raw_world_deg")
    fields += [
        "handedness_label",
        "handedness_score",
        "world_pose_jump_m",
        "flip_suspect",
        "flip_action",
    ]
    return fields


@contextmanager
def open_session_csv(csv_path, fields):
    """Open a session CSV and yield its writer and file handle."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        yield writer, csv_file


def write_measurement_row(writer, csv_file, row):
    """Append one measurement row and flush it to disk immediately."""
    writer.writerow(row)
    csv_file.flush()
