"""
pose_metrics.py
---------------
Three biomechanical metrics computed from SAM-3D keypoints.

Keypoint indices (from mhr_pose_info.py):
    5  = left_shoulder
    6  = right_shoulder
    9  = left_hip
    10 = right_hip
    11 = left_knee
    12 = right_knee
    41 = right_wrist
    62 = left_wrist

Coordinate convention (SAM-3D):
    x  → lateral (left/right)
    y  → vertical, but POINTING DOWN  (so -y is "up")
    z  → anterior/posterior (depth)

All functions accept:
    pts  : np.ndarray of shape (N, 3)  – the full keypoint array for one person
"""

import numpy as np


# ─── keypoint index constants ────────────────────────────────────────────────
IDX_LEFT_SHOULDER  = 5
IDX_RIGHT_SHOULDER = 6
IDX_LEFT_HIP       = 9
IDX_RIGHT_HIP      = 10
IDX_LEFT_KNEE      = 11
IDX_RIGHT_KNEE     = 12
IDX_RIGHT_WRIST    = 41
IDX_LEFT_WRIST     = 62


# ─── helper ──────────────────────────────────────────────────────────────────
def _unit(v):
    """Return the unit vector of v. Returns zero-vector if norm is ~0."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def _angle_between_deg(v1, v2):
    """Angle (degrees) between two vectors using the dot-product formula."""
    cos_theta = np.clip(np.dot(_unit(v1), _unit(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRUNK FLEXION ANGLE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_flexion(pts):
    """
    Compute trunk flexion angle.

    Method:
        mid_shoulder = average(left_shoulder, right_shoulder)
        mid_hip      = average(left_hip,      right_hip)
        trunk_vec    = mid_shoulder - mid_hip          (pointing upward along trunk)
        neg_y_axis   = [0, -1, 0]                     (true "up" in SAM-3D coords)
        flexion      = angle between trunk_vec and neg_y_axis  (degrees)

    Returns:
        flexion_deg  (float) – 0° = perfectly upright, >0° = leaning forward/backward
        trunk_vec    (np.ndarray shape (3,)) – the raw (non-unit) trunk vector
        mid_shoulder (np.ndarray shape (3,))
        mid_hip      (np.ndarray shape (3,))
    """
    mid_shoulder = (pts[IDX_LEFT_SHOULDER] + pts[IDX_RIGHT_SHOULDER]) / 2.0
    mid_hip      = (pts[IDX_LEFT_HIP]      + pts[IDX_RIGHT_HIP])      / 2.0

    trunk_vec  = mid_shoulder - mid_hip   # points from hips toward shoulders

    neg_y_axis = np.array([0.0, -1.0, 0.0])   # "up" in SAM-3D (y is down)

    flexion_deg = _angle_between_deg(trunk_vec, neg_y_axis)

    return flexion_deg, trunk_vec, mid_shoulder, mid_hip


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ASYMMETRY ANGLE  (pelvis vs. shoulder line)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_asymmetry(pts):
    """
    Compute lateral asymmetry between the hip and shoulder lines.

    Method:
        hip_vec      = left_hip - right_hip            (lateral hip axis)
        shoulder_vec = left_shoulder - right_shoulder  (lateral shoulder axis)
        asymmetry    = absolute angle between unit(hip_vec) and unit(shoulder_vec)

    A value of 0° means the two lines are perfectly parallel (no asymmetry).
    A non-zero value indicates pelvic/shoulder tilt mismatch.

    Returns:
        asymmetry_deg  (float)
        hip_unit       (np.ndarray shape (3,))
        shoulder_unit  (np.ndarray shape (3,))
    """
    hip_vec      = pts[IDX_LEFT_HIP]      - pts[IDX_RIGHT_HIP]
    shoulder_vec = pts[IDX_LEFT_SHOULDER] - pts[IDX_RIGHT_SHOULDER]

    hip_unit      = _unit(hip_vec)
    shoulder_unit = _unit(shoulder_vec)

    asymmetry_deg = abs(_angle_between_deg(hip_unit, shoulder_unit))

    return asymmetry_deg, hip_unit, shoulder_unit


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REACH DISTANCE IN CM  (mid-wrist to mid-shoulder, horizontal plane)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_reach_cm(pts, body_height_cm):
    """
    Convert keypoints to cm and compute horizontal reach distance.

    Conversion method:
        femur_length_ratio = 0.26   (femur / total body height)
        left_femur_px      = distance(left_hip,  left_knee)
        right_femur_px     = distance(right_hip, right_knee)
        avg_femur_px       = (left_femur_px + right_femur_px) / 2
        femur_length_cm    = body_height_cm * 0.26
        scale_cm_per_px    = femur_length_cm / avg_femur_px

    Reach distance:
        mid_wrist    = (right_wrist + left_wrist) / 2
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        reach_vec    = mid_wrist - mid_shoulder
        reach_vec_y0 = reach_vec with y-component set to 0  (horizontal plane)
        reach_cm     = norm(reach_vec_y0) * scale_cm_per_px

    Args:
        pts             : np.ndarray (N, 3) – keypoints in raw pixel/model units
        body_height_cm  : float – subject's body height in centimetres

    Returns:
        reach_cm        (float)  – signed horizontal reach in cm
                                   + = wrist is in front of shoulder (forward reach)
                                   - = wrist is behind shoulder (shoulder leads)
        scale           (float)  – cm-per-unit conversion factor
        mid_wrist       (np.ndarray shape (3,))
        mid_shoulder    (np.ndarray shape (3,))
        reach_vec_y0    (np.ndarray shape (3,)) – horizontal reach vector (y=0)
    """
    FEMUR_TO_HEIGHT_RATIO = 0.26

    # --- femur lengths in raw units ---
    left_femur_px  = np.linalg.norm(pts[IDX_LEFT_HIP]  - pts[IDX_LEFT_KNEE])
    right_femur_px = np.linalg.norm(pts[IDX_RIGHT_HIP] - pts[IDX_RIGHT_KNEE])
    avg_femur_px   = (left_femur_px + right_femur_px) / 2.0

    # --- scale factor ---
    femur_length_cm  = body_height_cm * FEMUR_TO_HEIGHT_RATIO
    scale_cm_per_unit = femur_length_cm / avg_femur_px   # cm per raw unit

    # --- mid-points ---
    mid_wrist    = (pts[IDX_RIGHT_WRIST]    + pts[IDX_LEFT_WRIST])     / 2.0
    mid_shoulder = (pts[IDX_LEFT_SHOULDER]  + pts[IDX_RIGHT_SHOULDER]) / 2.0
    mid_hip      = (pts[IDX_LEFT_HIP]       + pts[IDX_RIGHT_HIP])      / 2.0

    # --- horizontal reach vector (zero-out y component) ---
    reach_vec       = mid_wrist - mid_shoulder
    reach_vec_y0    = reach_vec.copy()
    reach_vec_y0[1] = 0.0                    # project onto horizontal (xz) plane

    # --- sign: forward direction is hip → shoulder (horizontal) ---
    #   positive  = wrist is in front of shoulder (further from hip)
    #   negative  = wrist is behind shoulder (shoulder is the furthest forward point)
    forward_vec    = mid_shoulder - mid_hip
    forward_vec[1] = 0.0                     # keep horizontal only
    forward_unit   = _unit(forward_vec)

    signed_magnitude = np.dot(reach_vec_y0, forward_unit)   # scalar, carries sign
    reach_cm = signed_magnitude * scale_cm_per_unit

    return reach_cm, scale_cm_per_unit, mid_wrist, mid_shoulder, reach_vec_y0


# ═══════════════════════════════════════════════════════════════════════════════
# Quick demo / sanity check
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json, os

    json_path = os.path.join("sam 3d body result", "keypoints_3d.json")
    if not os.path.exists(json_path):
        print(f"Cannot find {json_path}. Run sam3d_call() first.")
    else:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Support both {"people": [...]} and {"keypoints_3d": [...]} layouts
        if "people" in data:
            kp_list = [p["keypoints_3d"] for p in data["people"] if "keypoints_3d" in p]
        elif "keypoints_3d" in data:
            kp_list = [data["keypoints_3d"]]
        else:
            kp_list = [data]

        body_height_cm = 175.0   # ← change to subject's actual height

        for i, kp in enumerate(kp_list):
            pts = np.array(kp)
            print(f"\n── Person {i} ──────────────────────────────")

            # 1. Flexion
            flex, trunk_vec, mid_sh, mid_hp = compute_flexion(pts)
            print(f"  mid_shoulder : {mid_sh}")
            print(f"  mid_hip      : {mid_hp}")
            print(f"  trunk vector : {trunk_vec}")
            print(f"  Flexion      : {flex:.2f}°")

            # 2. Asymmetry
            asym, hip_u, sh_u = compute_asymmetry(pts)
            print(f"  hip unit vec : {hip_u}")
            print(f"  shl unit vec : {sh_u}")
            print(f"  Asymmetry    : {asym:.2f}°")

            # 3. Reach
            reach, scale, mid_wr, mid_sh2, rv0 = compute_reach_cm(pts, body_height_cm)
            print(f"  mid_wrist    : {mid_wr}")
            print(f"  mid_shoulder : {mid_sh2}")
            print(f"  reach vec xz : {rv0}")
            print(f"  Scale        : {scale:.4f} cm/unit")
            print(f"  Reach dist   : {reach:.2f} cm")

            
            