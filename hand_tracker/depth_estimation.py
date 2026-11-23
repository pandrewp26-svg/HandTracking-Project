"""Load Depth Anything V2 and run it without blocking the webcam loop."""

import sys
import threading
import time
import traceback
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


@dataclass(frozen=True)
class DepthResult:
    depth_map: np.ndarray
    source_frame: int
    source_timestamp: float
    inference_ms: float
    inference_fps: float
    source_image: np.ndarray
    source_landmarks: tuple


class DepthEstimator:
    """Load the official relative-depth model once and infer from BGR frames."""

    def __init__(self, repository_path, weights_path, input_size):
        """Load the relative-depth model and select its compute device."""
        repository_path = Path(repository_path).resolve()
        weights_path = Path(weights_path).resolve()
        package_path = repository_path / "depth_anything_v2"

        if not (package_path / "dpt.py").is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 source not found at {repository_path}. "
                "See README.md for setup."
            )
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found at {weights_path}. "
                "See README.md for setup."
            )
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required for Depth Anything V2. See README.md."
            ) from error

        repository = str(repository_path)
        if repository not in sys.path:
            sys.path.insert(0, repository)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as error:
            raise RuntimeError(
                "Could not import the official Depth Anything V2 implementation "
                f"from {repository_path}. Check its dependencies."
            ) from error

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.input_size = input_size
        self.torch = torch
        self.use_cuda_amp = (
            self.device.type == "cuda"
            and torch.cuda.get_device_capability(self.device)[0] >= 7
        )
        self.model = DepthAnythingV2(**MODEL_CONFIG)
        state_dict = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        print(
            "[Depth] Model loaded: Depth Anything V2 Small (vits), "
            f"device={self.device}, cuda_amp={self.use_cuda_amp}, "
            f"input_size={input_size}, weights={weights_path}",
            flush=True,
        )

    def infer(self, bgr_frame):
        """Run relative-depth inference for one BGR image."""
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        if bgr_frame.shape[0] == 0 or bgr_frame.shape[1] == 0:
            raise ValueError("Input image must have non-zero dimensions")
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            if self.use_cuda_amp
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            depth_map = np.asarray(
                self.model.infer_image(bgr_frame, self.input_size),
                dtype=np.float32,
            )
        if depth_map.size == 0:
            raise RuntimeError("Depth Anything V2 returned an empty depth map")
        if depth_map.shape != bgr_frame.shape[:2]:
            depth_map = cv2.resize(
                depth_map,
                (bgr_frame.shape[1], bgr_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if not np.isfinite(depth_map).any():
            raise RuntimeError("Depth Anything V2 returned no finite depth values")
        return depth_map


class DepthMapWorker:
    """Keep model inference off the camera loop and retain only the latest frame."""

    def __init__(self, repository_path, weights_path, input_size):
        """Load the estimator and start the background inference thread."""
        self.estimator = DepthEstimator(
            repository_path, weights_path, input_size=input_size
        )
        self._condition = threading.Condition()
        self._pending = None
        self._result = None
        self._error = None
        self._stopping = False
        self._submitted_count = 0
        self._replaced_count = 0
        self._inference_times = deque(maxlen=30)
        self._completion_times = deque(maxlen=30)
        self._thread = threading.Thread(
            target=self._run, name="depth-anything-v2", daemon=True
        )
        self._thread.start()
        print(
            f"[Depth] Worker started: alive={self._thread.is_alive()}",
            flush=True,
        )

    def submit(self, bgr_frame, frame_number, source_timestamp, landmarks):
        """Replace the pending frame and ask the worker to process it."""
        with self._condition:
            if self._stopping:
                return False
            # Replacing an unprocessed frame prevents inference backlog.
            if self._pending is not None:
                self._replaced_count += 1
            self._pending = (
                bgr_frame.copy(),
                frame_number,
                source_timestamp,
                tuple((float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks)
                if landmarks
                else (),
            )
            self._submitted_count += 1
            if self._submitted_count == 1 or self._submitted_count % 30 == 0:
                print(
                    "[Depth] Frame submitted: "
                    f"count={self._submitted_count}, frame={frame_number}, "
                    f"shape={bgr_frame.shape}, "
                    f"landmarks={len(landmarks)}, "
                    f"replaced_pending={self._replaced_count}, "
                    f"worker_alive={self._thread.is_alive()}",
                    flush=True,
                )
            self._condition.notify()
            return True

    def latest(self):
        """Return the newest completed depth result, if any."""
        with self._condition:
            return self._result

    @property
    def error(self):
        """Return the worker error that stopped depth inference."""
        with self._condition:
            return self._error

    def performance(self):
        """Return recent depth inference timing and backlog statistics."""
        with self._condition:
            average_ms = (
                sum(self._inference_times) / len(self._inference_times)
                if self._inference_times
                else 0.0
            )
            completion_fps = 0.0
            if len(self._completion_times) >= 2:
                duration = self._completion_times[-1] - self._completion_times[0]
                if duration > 0:
                    completion_fps = (
                        len(self._completion_times) - 1
                    ) / duration
            return {
                "input_size": self.estimator.input_size,
                "inference_fps": completion_fps,
                "average_inference_ms": average_ms,
                "submitted": self._submitted_count,
                "replaced_pending": self._replaced_count,
            }

    @property
    def is_alive(self):
        """Report whether the background inference thread is running."""
        return self._thread.is_alive()

    def close(self):
        """Stop the background worker and wait for its thread to finish."""
        print("[Depth] Stopping worker...", flush=True)
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join()
        print(
            f"[Depth] Worker stopped: alive={self._thread.is_alive()}",
            flush=True,
        )

    def _run(self):
        """Run pending depth jobs until the worker is closed or fails."""
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame, frame_number, source_timestamp, source_landmarks = self._pending
                self._pending = None

            started = time.perf_counter()
            print(
                f"[Depth] Inference started: frame={frame_number}",
                flush=True,
            )
            try:
                depth_map = self.estimator.infer(frame)
            except Exception as error:
                print(
                    f"[Depth] Inference failed: frame={frame_number}: {error}",
                    flush=True,
                )
                traceback.print_exc()
                with self._condition:
                    self._error = error
                    self._pending = None
                    self._stopping = True
                    self._condition.notify_all()
                return

            inference_ms = max((time.perf_counter() - started) * 1000.0, 1e-6)
            completed_at = time.perf_counter()
            print(
                "[Depth] Inference complete: "
                f"frame={frame_number}, elapsed_ms={inference_ms:.1f}, "
                f"shape={depth_map.shape}, min={float(depth_map.min()):.6f}, "
                f"max={float(depth_map.max()):.6f}, "
                f"mean={float(depth_map.mean()):.6f}, "
                f"landmarks={len(source_landmarks)}",
                flush=True,
            )
            with self._condition:
                self._inference_times.append(inference_ms)
                self._completion_times.append(completed_at)
                completion_fps = 0.0
                if len(self._completion_times) >= 2:
                    duration = (
                        self._completion_times[-1] - self._completion_times[0]
                    )
                    if duration > 0:
                        completion_fps = (
                            len(self._completion_times) - 1
                        ) / duration
                result = DepthResult(
                    depth_map=depth_map,
                    source_frame=frame_number,
                    source_timestamp=source_timestamp,
                    inference_ms=inference_ms,
                    inference_fps=completion_fps,
                    source_image=frame,
                    source_landmarks=source_landmarks,
                )
                self._result = result
