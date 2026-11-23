"""Render a fixed MuJoCo hand driven by finger bends and palm orientation."""

import cv2
import numpy as np

from .measurements import CONNECTIONS as HAND_CONNECTIONS


FINGER_COLORS = (
    "0.95 0.35 0.20 1",
    "0.20 0.65 1.00 1",
    "0.20 0.85 0.45 1",
    "1.00 0.70 0.20 1",
    "0.75 0.35 1.00 1",
)

# Easy-to-change raw-angle bend settings.
# Set FINGER_BEND_SCALE to reduce or increase every bend; set 0 to disable.
FINGER_BEND_SCALE = 1.0
FINGER_BEND_OFFSET_DEGREES = 0.0
BEND_DIRECTION_SIGN = 1.0
MAX_RAW_BEND_DEGREES = 120.0
PALM_DIR_SMOOTH = 0.25  # EMA fraction from the current wrist-to-middle ray.

# MediaPipe world: +X right, +Y down, +Z toward the camera.
# MuJoCo scene: +X right, +Y toward the camera, +Z up.
MEDIAPIPE_TO_MUJOCO = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])
assert np.isclose(np.linalg.det(MEDIAPIPE_TO_MUJOCO), 1.0)

# Static MuJoCo-space hand shape in meters. Live landmark positions do not
# move this template; only raw bend angles rotate each finger's distal points.
POINT_HAND_TEMPLATE = np.array([
    [0.000, 0.000, 0.145],  # 0 wrist
    [-0.025, 0.015, 0.160], # 1 thumb CMC
    [-0.045, 0.020, 0.180], # 2 thumb MCP
    [-0.060, 0.025, 0.205], # 3 thumb IP
    [-0.078, 0.030, 0.220], # 4 thumb tip
    [-0.035, -0.015, 0.195], # 5 index MCP
    [-0.038, -0.012, 0.245], # 6 index PIP
    [-0.040, -0.006, 0.280], # 7 index DIP
    [-0.042, 0.000, 0.305], # 8 index tip
    [-0.010, -0.005, 0.200], # 9 middle MCP
    [-0.010, 0.000, 0.255], # 10 middle PIP
    [-0.010, 0.006, 0.295], # 11 middle DIP
    [-0.010, 0.012, 0.325], # 12 middle tip
    [0.018, 0.010, 0.195],  # 13 ring MCP
    [0.020, 0.015, 0.245],  # 14 ring PIP
    [0.022, 0.021, 0.282],  # 15 ring DIP
    [0.024, 0.027, 0.308], # 16 ring tip
    [0.043, 0.025, 0.185], # 17 pinky MCP
    [0.050, 0.030, 0.228], # 18 pinky PIP
    [0.055, 0.036, 0.258], # 19 pinky DIP
    [0.060, 0.042, 0.280], # 20 pinky tip
], dtype=float)

HAND_PIVOT = np.mean(POINT_HAND_TEMPLATE[[0, 5, 9, 13, 17]], axis=0)


def mediapipe_vector_to_mujoco(vector):
    """Convert a MediaPipe vector (+X right, +Y down, +Z camera) to MuJoCo
    (+X right, +Y camera, +Z up), preserving its direction.
    """
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    vector = MEDIAPIPE_TO_MUJOCO @ vector
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-8 else None


def palm_orientation_rotation(target_normal, target_forward):
    """Build a proper hand rotation from palm-normal and wrist-to-middle rays."""
    def make_basis(normal, forward):
        normal = np.asarray(normal, dtype=float)
        forward = np.asarray(forward, dtype=float)
        normal_length = np.linalg.norm(normal)
        if normal.shape != (3,) or forward.shape != (3,):
            return None
        if not np.isfinite(normal).all() or not np.isfinite(forward).all():
            return None
        if normal_length < 1e-8:
            return None
        normal = normal / normal_length
        forward = forward - np.dot(forward, normal) * normal
        forward_length = np.linalg.norm(forward)
        if forward_length < 1e-8:
            return None
        forward /= forward_length
        side = np.cross(normal, forward)
        return np.column_stack((forward, side, normal))

    wrist = POINT_HAND_TEMPLATE[0]
    template_normal = np.cross(
        POINT_HAND_TEMPLATE[5] - wrist,
        POINT_HAND_TEMPLATE[17] - wrist,
    )
    template_forward = POINT_HAND_TEMPLATE[9] - wrist
    source_basis = make_basis(template_normal, template_forward)
    target_basis = make_basis(target_normal, target_forward)
    if source_basis is None or target_basis is None:
        return None

    rotation = target_basis @ source_basis.T
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    return rotation


def apply_raw_finger_bends(points, raw_angles):
    """Curl the fixed template's distal finger segments using raw angles."""
    bent = np.asarray(points, dtype=float).copy()
    if bent.shape != (21, 3):
        return bent

    # Each tuple is (hinge landmark, distal landmarks).
    # Thumb ends at landmark 4; landmark 5 belongs to the index finger.
    finger_joints = {
        "Thumb": (3, (4,)),
        "Index": (6, (7, 8)),
        "Middle": (10, (11, 12)),
        "Ring": (14, (15, 16)),
        "Pinky": (18, (19, 20)),
    }

    for name, (hinge, distal_points) in finger_joints.items():
        raw_angle = raw_angles.get(name)
        if raw_angle is None:
            continue

        bend_degrees = float(np.clip(
            float(raw_angle) * FINGER_BEND_SCALE
            + FINGER_BEND_OFFSET_DEGREES,
            0.0,
            MAX_RAW_BEND_DEGREES,
        ))
        angle = np.radians(
            -bend_degrees * BEND_DIRECTION_SIGN
        )
        if abs(angle) < 1e-8:
            continue

        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotate_about_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ])
        pivot = bent[hinge].copy()
        for point_index in distal_points:
            bent[point_index] = (
                pivot
                + rotate_about_x @ (bent[point_index] - pivot)
            )

    return bent


class MuJoCoHandViewer:
    """Render the existing 21-point hand plus floor and orbit camera."""

    def __init__(self):
        """Create the scene, renderer, orbit camera, and point mocap bodies."""
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "MuJoCo is not installed. Run python -m pip install mujoco."
            ) from error

        self.mujoco = mujoco
        sphere_bodies = "".join(
            f'''
            <body name="landmark_{index:02d}" mocap="true">
              <geom name="landmark_{index:02d}_geom" type="sphere"
                    size="0.008" rgba="{landmark_rgba(index)}"/>
            </body>
            '''
            for index in range(21)
        )
        bone_bodies = "".join(
            f'''
            <body name="bone_{index:02d}" mocap="true">
              <geom name="bone_{index:02d}_geom" type="capsule"
                    size="0.004 0.01" rgba="{bone_rgba(index)}"/>
            </body>
            '''
            for index in range(len(HAND_CONNECTIONS))
        )
        xml = f'''
        <mujoco model="hand_view">
          <option gravity="0 0 -9.81"/>
          <visual>
            <headlight ambient="0.25 0.25 0.25"
                       diffuse="0.75 0.75 0.75"
                       specular="0.30 0.30 0.30"/>
            <global offwidth="800" offheight="600"/>
            <rgba haze="0.75 0.82 0.90 1"/>
          </visual>
          <asset>
            <texture name="checker" type="2d" builtin="checker"
                     rgb1="0.75 0.75 0.75" rgb2="0.28 0.28 0.28"
                     width="128" height="128"/>
            <material name="floor" texture="checker" texrepeat="8 8"
                      reflectance="0.10" rgba="1 1 1 1"/>
          </asset>
          <worldbody>
            <light name="key" pos="0.7 0.5 1.2" dir="-0.5 -0.3 -1"
                   directional="true" diffuse="1 1 1"/>
            <geom name="floor" type="plane" size="2 2 0.05"
                  material="floor"/>
            <geom name="axis_x" type="capsule" fromto="0 0 0.005 0.16 0 0.005"
                  size="0.008" rgba="1 0.1 0.1 1"/>
            <geom name="axis_y" type="capsule" fromto="0 0 0.005 0 0.16 0.005"
                  size="0.008" rgba="0.1 1 0.1 1"/>
            <geom name="axis_z" type="capsule" fromto="0 0 0.005 0 0 0.16"
                  size="0.008" rgba="0.1 0.3 1 1"/>
            {sphere_bodies}
            {bone_bodies}
          </worldbody>
        </mujoco>
        '''

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=600, width=800)
        self.camera = mujoco.MjvCamera()
        self.reset_camera()

        self.sphere_mocap_ids = [
            get_mocap_id(self.model, mujoco, "body", f"landmark_{index:02d}")
            for index in range(21)
        ]
        self.bone_mocap_ids = [
            get_mocap_id(self.model, mujoco, "body", f"bone_{index:02d}")
            for index in range(len(HAND_CONNECTIONS))
        ]
        self.last_raw_angles = {
            "Thumb": None,
            "Index": None,
            "Middle": None,
            "Ring": None,
            "Pinky": None,
        }
        self.last_palm_normal = None
        self.last_palm_direction = None
        self.last_orientation = np.eye(3)
        self.show_hud = False
        self.dragging = False
        self.last_mouse = None
        self.window_name = "MuJoCo 3D Hand"

    def reset_camera(self):
        """Reset the explicit orbit camera."""
        self.camera.lookat[:] = [0.0, 0.0, 0.18]
        self.camera.distance = 0.55
        self.camera.azimuth = 90
        self.camera.elevation = -12

    def set_mouse_callback(self):
        """Enable drag orbit and wheel zoom."""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, parameter):
        del parameter
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.last_mouse is not None:
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]
                self.camera.azimuth += dx * 0.35
                self.camera.elevation = float(np.clip(
                    self.camera.elevation + dy * 0.25,
                    -89.0,
                    89.0,
                ))
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_direction = 1.0 if flags > 0 else -1.0
            zoom_factor = 1.0 - wheel_direction * 0.08
            self.camera.distance = float(np.clip(
                self.camera.distance * zoom_factor,
                0.18,
                2.0,
            ))

    def handle_key(self, key):
        """Handle only free MuJoCo camera/HUD keys."""
        if key == ord("["):
            self.camera.azimuth -= 5
        elif key == ord("]"):
            self.camera.azimuth += 5
        elif key == ord(","):
            self.camera.elevation = float(np.clip(
                self.camera.elevation - 5,
                -89,
                89,
            ))
        elif key == ord("."):
            self.camera.elevation = float(np.clip(
                self.camera.elevation + 5,
                -89,
                89,
            ))
        elif key == ord("-"):
            self.camera.distance = float(np.clip(
                self.camera.distance * 1.08,
                0.18,
                2.0,
            ))
        elif key == ord("+") or key == ord("="):
            self.camera.distance = float(np.clip(
                self.camera.distance / 1.08,
                0.18,
                2.0,
            ))
        elif key == ord("0"):
            self.reset_camera()
        elif key == ord("u"):
            self.show_hud = not self.show_hud
        else:
            return False
        return True

    def update(self, raw_angles=None, palm_normal=None, palm_direction=None):
        """Update curls, then orient the hand using its normal and forward ray."""
        if raw_angles is None:
            raw_angles = {}
        for name in self.last_raw_angles:
            angle = raw_angles.get(name)
            if angle is not None:
                self.last_raw_angles[name] = angle
        raw_angles = self.last_raw_angles.copy()

        points = apply_raw_finger_bends(POINT_HAND_TEMPLATE, raw_angles)
        converted_normal = (
            mediapipe_vector_to_mujoco(palm_normal)
            if palm_normal is not None
            else None
        )
        converted_direction = (
            mediapipe_vector_to_mujoco(palm_direction)
            if palm_direction is not None
            else None
        )
        if converted_normal is not None:
            self.last_palm_normal = converted_normal
        if converted_direction is not None:
            if self.last_palm_direction is None:
                self.last_palm_direction = converted_direction
            else:
                smoothed = (
                    (1.0 - PALM_DIR_SMOOTH) * self.last_palm_direction
                    + PALM_DIR_SMOOTH * converted_direction
                )
                length = np.linalg.norm(smoothed)
                self.last_palm_direction = (
                    smoothed / length
                    if length > 1e-8
                    else converted_direction
                )
        if self.last_palm_normal is not None and self.last_palm_direction is not None:
            orientation = palm_orientation_rotation(
                self.last_palm_normal,
                self.last_palm_direction,
            )
            if orientation is not None:
                self.last_orientation = orientation
        points = (
            (points - HAND_PIVOT) @ self.last_orientation.T
            + HAND_PIVOT
        )

        for mocap_id, point in zip(self.sphere_mocap_ids, points):
            self.data.mocap_pos[mocap_id] = point
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

        for bone_index, (first, second) in enumerate(HAND_CONNECTIONS):
            start = points[first]
            end = points[second]
            midpoint = (start + end) / 2.0
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 1e-8:
                continue
            direction /= length

            mocap_id = self.bone_mocap_ids[bone_index]
            self.data.mocap_pos[mocap_id] = midpoint
            self.data.mocap_quat[mocap_id] = direction_to_quaternion(direction)
            geom_id = get_mocap_id(
                self.model,
                self.mujoco,
                "geom",
                f"bone_{bone_index:02d}_geom",
            )
            self.model.geom_size[geom_id, 1] = length / 2.0

        self.mujoco.mj_forward(self.model, self.data)

    def render(self):
        """Render RGB through MuJoCo and convert to OpenCV BGR."""
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
        image = rgb[:, :, ::-1].copy()
        if self.show_hud:
            draw_raw_angle_hud(image, self.last_raw_angles)
        cv2.putText(
            image,
            "[/] az  [,/.] el  +/- zoom  0 reset  U HUD",
            (12, image.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return image

    def close(self):
        """Release the MuJoCo renderer."""
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None


def get_mocap_id(model, mujoco, object_kind, name):
    """Get a named MuJoCo body or geom ID."""
    if object_kind == "body":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return model.body_mocapid[object_id]
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def draw_raw_angle_hud(image, raw_angles):
    """Draw the current raw finger bend angles on the 3D window."""
    lines = ["Raw finger bend angles (degrees)"]
    for name, value in raw_angles.items():
        text = "Hold" if value is None else f"{float(value):.1f}"
        lines.append(f"{name}: {text}")
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (20, 20, 20),
            3,
        )
        cv2.putText(
            image,
            line,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )


def landmark_rgba(index):
    """Return a finger-specific landmark color."""
    if index == 0:
        return "0.95 0.95 0.95 1"
    if 1 <= index <= 4:
        return FINGER_COLORS[0]
    if 5 <= index <= 8:
        return FINGER_COLORS[1]
    if 9 <= index <= 12:
        return FINGER_COLORS[2]
    if 13 <= index <= 16:
        return FINGER_COLORS[3]
    return FINGER_COLORS[4]


def bone_rgba(index):
    """Return a color for a point-skeleton bone."""
    first = HAND_CONNECTIONS[index][0]
    if first == 0:
        return "0.85 0.85 0.85 1"
    if first in (1, 2, 3):
        return FINGER_COLORS[0]
    if first in (5, 6, 7):
        return FINGER_COLORS[1]
    if first in (9, 10, 11):
        return FINGER_COLORS[2]
    if first in (13, 14, 15):
        return FINGER_COLORS[3]
    return FINGER_COLORS[4]


def direction_to_quaternion(direction):
    """Rotate a capsule Z axis onto a bone direction."""
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -1.0 + 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    axis = np.cross(z_axis, direction)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return [
        float(np.cos(angle / 2.0)),
        float(axis[0] * np.sin(angle / 2.0)),
        float(axis[1] * np.sin(angle / 2.0)),
        float(axis[2] * np.sin(angle / 2.0)),
    ]
