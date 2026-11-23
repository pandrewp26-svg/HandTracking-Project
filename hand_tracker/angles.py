"""Store the latest raw and filtered angle values for other modules."""

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
AXIS_NAMES = ("x", "y", "z")

finger_angles = {
    name: {"raw": None, "filtered": None}
    for name in FINGER_NAMES
}

palm_normal_angles = {
    name: {"raw": None, "filtered": None}
    for name in AXIS_NAMES
}


def reset_angles():
    """Clear every stored raw and filtered angle."""
    for values in (*finger_angles.values(), *palm_normal_angles.values()):
        values["raw"] = None
        values["filtered"] = None


def get_angle(group, name, kind="raw"):
    """Read a finger or palm angle by name and raw/filtered kind."""
    if group == "finger":
        angles = finger_angles
    elif group == "palm":
        angles = palm_normal_angles
    else:
        raise ValueError(f"Unknown angle group: {group}")
    return angles[name][kind]
