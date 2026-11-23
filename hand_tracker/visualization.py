"""Draw camera overlays and save measurement graphs."""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .measurements import CONNECTIONS


def draw_hand_connections(frame, points_2d):
    """Draw the MediaPipe hand skeleton on a camera frame."""
    for a, b in CONNECTIONS:
        cv2.line(
            frame,
            tuple(points_2d[a].astype(int)),
            tuple(points_2d[b].astype(int)),
            (0, 220, 0), 2,
        )


def draw_palm_normal(frame, normal):
    """Draw the palm-normal X, Y, and Z components on the frame."""
    cv2.putText(
        frame,
        f"Normal X:{normal[0]:.2f} Y:{normal[1]:.2f} Z:{normal[2]:.2f}",
        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 2,
    )


def draw_index_bend(frame, raw):
    """Draw the raw index-bend angle on the camera frame."""
    cv2.putText(
        frame, f"Index bend: {raw:.1f} deg",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2,
    )


def draw_trial_status(frame, trial, frame_number):
    """Draw the current trial number and frame number."""
    cv2.putText(
        frame, f"Trial: {trial} | Frame: {frame_number}",
        (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def draw_depth_status(
    frame, enabled, worker_alive, result, now, error=None,
    stale_after_ms=2500,
):
    """Make depth initialization, readiness, and visualization state visible."""
    if error is not None:
        message = f"Depth unavailable: {error}"
        color = (0, 0, 255)
    elif not worker_alive:
        message = "Depth worker stopped (see terminal for details)"
        color = (0, 0, 255)
    elif result is None:
        message = "Depth inference pending; press D to toggle depth view"
        color = (0, 200, 255)
    else:
        age_ms = max(now - result.source_timestamp, 0.0) * 1000.0
        age = age_ms / 1000.0
        age_text = f"{age:.1f}s"
        freshness = "STALE" if age_ms > stale_after_ms else "current"
        if result.source_landmarks:
            message = (
                f"Depth {freshness}: frame {result.source_frame}, age {age_text}; "
                f"{'view ON' if enabled else 'press D for view'}"
            )
            color = (0, 200, 255) if age_ms > stale_after_ms else (0, 255, 0)
        else:
            message = (
                f"Depth map {freshness}: frame {result.source_frame}, age {age_text}; "
                f"no hand in source frame"
            )
            color = (0, 200, 255)

    cv2.putText(
        frame, message, (12, 118), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2,
    )


def draw_performance_overlay(
    frame,
    camera_fps,
    mediapipe_fps,
    dav2_fps,
    average_depth_ms,
    depth_age_ms,
    depth_input_size,
    stale=False,
):
    """Draw compact rolling pipeline performance metrics on the camera feed."""
    age = "n/a" if depth_age_ms is None else f"{depth_age_ms:.0f} ms"
    line_one = (
        f"Cam {camera_fps:.1f} FPS | MP {mediapipe_fps:.1f} FPS | "
        f"DAV2 {dav2_fps:.2f} FPS, avg {average_depth_ms:.0f} ms"
    )
    line_two = (
        f"DAV2 input {depth_input_size}px | depth age {age}"
        f"{' STALE' if stale else ''}"
    )
    cv2.putText(
        frame, line_one, (12, 144), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, line_two, (12, 164), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 2,
    )


def show_depth_comparison(
    frame,
    depth_map,
    points_2d=None,
    landmark_depths=None,
    source_frame=None,
    age_frames=None,
    inference_ms=None,
    inference_fps=None,
    average_inference_ms=None,
    input_size=None,
    landmark_validity=None,
    stale=False,
):
    """Display RGB and relative-depth views; colormap scaling is visualization-only."""
    original_height, original_width = depth_map.shape
    frame_height, frame_width = frame.shape[:2]
    if depth_map.shape != (frame_height, frame_width):
        depth_map = cv2.resize(
            depth_map,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if points_2d is not None:
            points_2d = np.asarray(points_2d, dtype=float).copy()
            points_2d[:, 0] *= (
                (frame_width - 1) / max(original_width - 1, 1)
            )
            points_2d[:, 1] *= (
                (frame_height - 1) / max(original_height - 1, 1)
            )

    finite = np.isfinite(depth_map)
    if finite.any():
        low, high = np.percentile(depth_map[finite], [2, 98])
    else:
        low = high = 0.0
    if high > low:
        scaled = np.clip((depth_map - low) / (high - low), 0.0, 1.0)
    else:
        scaled = np.zeros_like(depth_map)
    scaled[~finite] = 0.0
    depth_view = cv2.applyColorMap(
        (scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    rgb_view = frame.copy()

    if points_2d is not None:
        for index, point in enumerate(points_2d):
            x = int(np.clip(point[0], 0, depth_view.shape[1] - 1))
            y = int(np.clip(point[1], 0, depth_view.shape[0] - 1))
            cv2.circle(rgb_view, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(depth_view, (x, y), 3, (255, 255, 255), -1)
            valid = (
                landmark_depths is not None
                and landmark_validity is not None
                and landmark_validity[index]
            )
            label = (
                f"{index}:{landmark_depths[index]:.2f}"
                if valid
                else f"{index}:NA"
            )
            for view in (rgb_view, depth_view):
                position = (
                    min(x + 4, max(view.shape[1] - 70, 0)),
                    max(y - 4, 12),
                )
                cv2.putText(
                    view, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 0, 0), 3,
                )
                cv2.putText(
                    view, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1,
                )

    cv2.putText(
        rgb_view, "RGB camera frame", (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
    )
    status = "Depth Anything V2 relative depth"
    if source_frame is not None:
        status += f" | source frame {source_frame}"
    if age_frames is not None:
        status += f" | age {age_frames} frames"
    if inference_ms is not None and inference_fps is not None:
        status += f" | completion {inference_fps:.2f} FPS"
    if average_inference_ms is not None:
        status += f" | avg {average_inference_ms:.0f} ms"
    if input_size is not None:
        status += f" | input {input_size}px"
    if stale:
        status += " | STALE"
    cv2.putText(
        depth_view, status, (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    cv2.imshow(
        "Depth Anything V2 comparison",
        np.hstack((rgb_view, depth_view)),
    )


def save_plots(bend_history, normal_history, plot_dir, session, stamp):
    """Save the index-bend and palm-normal histories as PNG graphs."""
    plots = [
        (bend_history, ["Raw"],
         ["Index bend (degrees)", "Raw 3D Index Bend"], "bend"),
        (normal_history, ["X angle", "Y angle", "Z angle"],
         ["Angle to positive axis (degrees)", "Approximate Palm-Normal Axis Angles"],
         "normal"),
    ]

    for history, labels, text, suffix in plots:
        if not history:
            continue

        data = np.array(history)
        plt.figure(figsize=(10, 4 if suffix == "bend" else 5))

        for i, label in enumerate(labels, start=1):
            plt.plot(data[:, 0], data[:, i], label=label)

        plt.xlabel("Frame")
        plt.ylabel(text[0])
        plt.title(text[1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = plot_dir / f"session_{session:03d}_{stamp}_{suffix}.png"
        plt.savefig(path, dpi=150)
        plt.close()


def _live_ticks(lower, upper):
    span = upper - lower
    step = span / 6
    magnitude = 10 ** np.floor(np.log10(step))
    fraction = step / magnitude
    nice_fraction = next(
        value for value in (1, 2, 2.5, 5, 10) if fraction <= value
    )
    step = nice_fraction * magnitude
    first = np.ceil(lower / step) * step
    return np.arange(first, upper + step * 0.5, step)


def _draw_live_panel(image, data, title, series, top, bottom):
    left, right = 78, 782
    plot_data = data[:, 1:] if data.size else np.empty((0, len(series)))
    if data.size:
        x_lower, x_upper = data[:, 0].min(), data[:, 0].max()
        y_lower, y_upper = plot_data.min(), plot_data.max()
        if x_lower == x_upper:
            x_lower, x_upper = x_lower - 0.5, x_upper + 0.5
        if y_lower == y_upper:
            padding = max(abs(y_lower) * 0.05, 0.05)
            y_lower, y_upper = y_lower - padding, y_upper + padding
        x_padding = (x_upper - x_lower) * 0.05
        y_padding = (y_upper - y_lower) * 0.05
        x_lower, x_upper = x_lower - x_padding, x_upper + x_padding
        y_lower, y_upper = y_lower - y_padding, y_upper + y_padding
    else:
        x_lower, x_upper, y_lower, y_upper = 0, 1, 0, 1

    x_ticks = _live_ticks(x_lower, x_upper)
    y_ticks = _live_ticks(y_lower, y_upper)
    for tick in x_ticks:
        x = int(left + (tick - x_lower) * (right - left) / (x_upper - x_lower))
        cv2.line(image, (x, top), (x, bottom), (228, 228, 228), 1, cv2.LINE_AA)
        label = f"{tick:g}"
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(
            image, label, (x - width // 2, bottom + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
    for tick in y_ticks:
        y = int(bottom - (tick - y_lower) * (bottom - top) / (y_upper - y_lower))
        cv2.line(image, (left, y), (right, y), (228, 228, 228), 1, cv2.LINE_AA)
        label = f"{tick:g}"
        width, height = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )[0]
        cv2.putText(
            image, label, (left - width - 9, y + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 1)
    title_width = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1
    )[0][0]
    title_x = left + (right - left - title_width) // 2
    cv2.putText(
        image, title, (title_x, top - 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "Frame", (left + (right - left - 48) // 2, bottom + 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
    )
    label = "Degrees"
    label_width, label_height = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )[0]
    label_image = np.full((label_width + 4, label_height + 8, 3), 255, dtype=np.uint8)
    cv2.putText(
        label_image, label, (4, label_width + 1), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 0), 1, cv2.LINE_AA,
    )
    label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y_label = (top + bottom - label_image.shape[0]) // 2
    image[y_label:y_label + label_image.shape[0], 24:24 + label_image.shape[1]] = label_image

    if data.size:
        for column, label, color in series:
            points = np.column_stack((
                left + (data[:, 0] - x_lower) * (right - left) / (x_upper - x_lower),
                bottom - (data[:, column] - y_lower) * (bottom - top) / (y_upper - y_lower),
            )).astype(np.int32)
            cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)

        legend_left = right - 104
        legend_top = top + 8
        legend_height = 22 * len(series) + 8
        cv2.rectangle(
            image, (legend_left, legend_top),
            (right - 5, legend_top + legend_height),
            (255, 255, 255), -1,
        )
        cv2.rectangle(
            image, (legend_left, legend_top),
            (right - 5, legend_top + legend_height),
            (190, 190, 190), 1,
        )
        for index, (_, label, color) in enumerate(series):
            y = legend_top + 17 + 22 * index
            cv2.line(
                image, (legend_left + 8, y - 4), (legend_left + 29, y - 4),
                color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                image, label, (legend_left + 36, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
            )

def render_live_graphs(bend_history, normal_history):
    """Create a two-panel live graph image for optional OpenCV display."""
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    bend_data = np.asarray(bend_history)
    normal_data = np.asarray(normal_history)
    _draw_live_panel(
        image, bend_data, "Index Bend",
        [(1, "Raw", (180, 119, 31))], 42, 255,
    )
    _draw_live_panel(
        image, normal_data, "Palm Normal Angles",
        [
            (1, "X", (180, 119, 31)),
            (2, "Y", (14, 127, 255)),
            (3, "Z", (44, 160, 44)),
        ], 342, 555,
    )
    return image
