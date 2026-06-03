"""
lifting_analysis_utils.py
--------------------------
Helper functions for the lifting image analysis pipeline.

Pipeline:
    1. extract_body_params_from_prompt()  – parse BH/BW from user text via GPT
    2. estimate_object_weight()           – Vision call to estimate handled object weight (M)
    3. _plot_three_panel()                – 3-panel figure: original image | 3-D skeleton | PLY mesh
    4. run_full_lifting_analysis()        – orchestrates pose metrics + spinal load computation
"""

import os
import sys
import glob
import json
import base64
import io
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in threaded Streamlit)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from openai import OpenAI

# ── Make sure parent packages on the path ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pose_metrics import (
    compute_flexion,
    compute_asymmetry,
    compute_reach_cm,
)
from spinal_load import compute_spinal_loads_standing, compute_spinal_loads_flex
from mhr_pose_info import pose_info


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract body height & weight from user prompt
# ─────────────────────────────────────────────────────────────────────────────
def extract_body_params_from_prompt(prompt: str, client: OpenAI) -> dict:
    """
    Ask GPT to extract body height (cm) and body weight (kg) from the prompt.
    Returns {"BH": float, "BW": float} — defaults 175 / 75 if not mentioned.
    """
    system_msg = (
        "You are a precise data-extraction assistant. "
        "Your ONLY task is to find body height and body weight values in the user's text.\n\n"
        "Rules:\n"
        "- Look for patterns like '175 cm', '5 foot 11', '80 kg', '176 pounds', etc.\n"
        "- Convert feet/inches to centimetres: 1 foot = 30.48 cm, 1 inch = 2.54 cm.\n"
        "- Convert pounds to kilograms: 1 lb = 0.4536 kg.\n"
        "- If height is NOT mentioned → output BH = 175.0\n"
        "- If weight is NOT mentioned → output BW = 75.0\n"
        "- Output ONLY valid JSON with exactly two keys: {\"BH\": <float_cm>, \"BW\": <float_kg>}"
    )
    user_msg = (
        f"Extract body height and body weight from this text:\n\n\"{prompt}\"\n\n"
        "If neither is mentioned, return the defaults (BH=175, BW=75)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        BH = float(data.get("BH", 175))
        BW = float(data.get("BW", 75))
    except Exception as e:
        print(f"[extract_body_params] GPT error, using defaults: {e}")
        BH, BW = 175.0, 75.0

    # Sanity-check ranges
    BH = BH if 100 < BH < 250 else 175.0
    BW = BW if 30  < BW < 300 else 75.0
    return {"BH": BH, "BW": BW}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Estimate handled-object weight from image (Vision)
# ─────────────────────────────────────────────────────────────────────────────
def _encode_image(image_path: str) -> tuple:
    """Return (base64_string, mime_type) for an image file."""
    ext  = os.path.splitext(image_path)[-1].lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "tiff": "image/tiff",
            "bmp": "image/bmp",  "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def estimate_object_weight(image_path: str, client: OpenAI) -> float:
    """
    Send image to OpenAI Vision (gpt-4.1) and estimate the weight (kg)
    of the object being handled.  Uses a detailed, opinionated prompt so
    the model cannot escape with a zero answer when an object is clearly
    visible.  Returns a float (defaults to 5.0 if truly undetermined).
    """
    try:
        b64, mime = _encode_image(image_path)

        weight_prompt = (
            "You are a biomechanics assistant estimating the weight of a handled object for spinal load analysis.\n\n"
            "Look carefully at the object the person is holding, carrying, or lifting in this image.\n\n"
            "Steps to follow:\n"
            "1. Identify WHAT the object is (box, package, tool, weight plate, bag, etc.).\n"
            "2. Estimate its dimensions relative to the person's body.\n"
            "3. Apply typical density / real-world weight knowledge:\n"
            "   - Small cardboard box (30×20×20 cm): ~5 kg\n"
            "   - Medium cardboard box (50×40×30 cm): ~10–15 kg\n"
            "   - Large heavy box: ~20–30 kg\n"
            "   - Weight plate (45 cm diameter): ~20 kg\n"
            "   - Tool / small object: ~1–3 kg\n"
            "4. You MUST give a best-estimate number. Do NOT say 0 unless the person's hands are completely empty.\n"
            "5. Reply with ONLY a single numeric value in kilograms, no units, no explanation.\n"
            "   Examples of valid replies: 5  or  12.5  or  3.0"
        )

        response = client.chat.completions.create(
            model="gpt-4.1",          # best available vision model
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": weight_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}",
                                   "detail": "high"}},
                ],
            }],
            max_tokens=15,
        )
        raw = response.choices[0].message.content.strip()
        print(f"[estimate_object_weight] raw response: '{raw}'")
        # Keep only digits and the first decimal point
        cleaned = "".join(c for c in raw if c.isdigit() or c == ".")
        val = float(cleaned) if cleaned else 5.0
        return val if val > 0 else 5.0   # never return 0 from a non-empty response

    except Exception as e:
        print(f"[estimate_object_weight] Vision error: {e}")
        return 5.0   # conservative non-zero default


# ─────────────────────────────────────────────────────────────────────────────
# 3. Three-panel figure: original image | 3-D skeleton | PLY mesh
# ─────────────────────────────────────────────────────────────────────────────
def _draw_skeleton(ax, pts: np.ndarray):
    """Draw the 3-D skeleton onto an existing Axes3D."""
    skeleton      = pose_info.get("skeleton_info", {})
    keypoint_info = pose_info.get("keypoint_info", {})
    name_to_idx   = {v["name"]: v["id"] for v in keypoint_info.values()
                     if "name" in v and "id" in v}

    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
               c="dodgerblue", s=20, zorder=5, depthshade=False)

    for v in skeleton.values():
        n1, n2 = v["link"]
        if n1 in name_to_idx and n2 in name_to_idx:
            i1, i2 = name_to_idx[n1], name_to_idx[n2]
            if i1 < len(pts) and i2 < len(pts):
                color = np.array(v.get("color", [0, 0, 0])) / 255.0
                ax.plot([pts[i1, 0], pts[i2, 0]],
                        [pts[i1, 2], pts[i2, 2]],
                        [-pts[i1, 1], -pts[i2, 1]],
                        color=color, linewidth=1.5)

    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Z", fontsize=8)
    ax.set_zlabel("-Y", fontsize=8)
    ax.set_title("3-D Keypoints (SAM-3D)", fontsize=10)

    # Equal aspect ratio
    lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    r    = max(abs(h - l) for l, h in lims) / 2
    mids = [(l + h) / 2 for l, h in lims]
    ax.set_xlim3d([mids[0] - r, mids[0] + r])
    ax.set_ylim3d([mids[1] - r, mids[1] + r])
    ax.set_zlim3d([mids[2] - r, mids[2] + r])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _draw_ply(ax, result_dir: str):
    """Find the first .ply file in result_dir and render it as a point cloud."""
    ply_files = glob.glob(os.path.join(result_dir, "*.ply"))
    if not ply_files:
        ax.text(0.5, 0.5, 0.5, "No .ply file found",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_title("3-D Mesh (PLY)", fontsize=10)
        return

    ply_path = ply_files[0]
    try:
        # Try plyfile first (lightweight)
        from plyfile import PlyData
        ply  = PlyData.read(ply_path)
        verts = ply["vertex"]
        x, y, z = np.array(verts["x"]), np.array(verts["y"]), np.array(verts["z"])
        # Sample for performance (max 30 000 pts)
        if len(x) > 30_000:
            idx = np.random.choice(len(x), 30_000, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        # Use vertex colors if available
        try:
            r = np.array(verts["red"])   / 255.0
            g = np.array(verts["green"]) / 255.0
            b = np.array(verts["blue"])  / 255.0
            colors = np.stack([r, g, b], axis=1)
            if len(x) > 30_000:
                colors = colors[idx]
        except Exception:
            colors = "steelblue"
        ax.scatter(x, z, -y, c=colors, s=0.5, alpha=0.6)
    except ImportError:
        # Fallback: read PLY manually (binary or ascii)
        _draw_ply_manual(ax, ply_path)
    except Exception as e:
        ax.text2D(0.5, 0.5, f"PLY read error:\n{e}",
                  ha="center", va="center", fontsize=7,
                  transform=ax.transAxes)

    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Z", fontsize=8)
    ax.set_zlabel("-Y", fontsize=8)
    ax.set_title(f"3-D Mesh — {os.path.basename(ply_path)}", fontsize=9)


def _draw_ply_manual(ax, ply_path: str):
    """Minimal manual PLY ASCII reader as fallback (no plyfile dependency)."""
    xs, ys, zs = [], [], []
    in_vertex = False
    vertex_count = 0
    header_done = False
    with open(ply_path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not header_done:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line == "end_header":
                    header_done = True
                    in_vertex = True
                continue
            if in_vertex and vertex_count > 0:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
                        zs.append(float(parts[2]))
                    except ValueError:
                        pass
                    vertex_count -= 1
    if xs:
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        if len(xs) > 30_000:
            idx = np.random.choice(len(xs), 30_000, replace=False)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
        ax.scatter(xs, zs, -ys, s=0.5, c="steelblue", alpha=0.6)


def _plot_two_panel(pts: np.ndarray, image_path: str) -> bytes:
    """
    Build a 2-panel matplotlib figure and return as PNG bytes.

    Panel 1 — Original uploaded image
    Panel 2 — 3-D keypoints skeleton
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("SAM-3D Lifting Analysis", fontsize=13, fontweight="bold", y=1.01)

    # ── Panel 1: Original image ───────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    try:
        img = mpimg.imread(image_path)
        ax1.imshow(img)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Image load error:\n{e}", ha="center", va="center",
                 fontsize=8, transform=ax1.transAxes)
    ax1.set_title("Original Image", fontsize=10)
    ax1.axis("off")

    # ── Panel 2: 3-D skeleton ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_skeleton(ax2, pts)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive PLY viewer  (Plotly Mesh3d)
# ─────────────────────────────────────────────────────────────────────────────
def build_ply_plotly_fig(result_dir: str):
    """
    Find the first .ply file in result_dir, read vertices + faces + vertex
    colours (via plyfile), and return a Plotly Figure with go.Mesh3d for
    interactive rotation/zoom in Streamlit.

    Coordinate convention: SAM-3D has y pointing down, so we map
        x → x,  z → y-axis of plot,  -y → z-axis of plot
    so the figure looks upright.

    Returns None if no PLY is found or loading fails.
    """
    import plotly.graph_objects as go

    ply_files = glob.glob(os.path.join(result_dir, "*.ply"))
    if not ply_files:
        print("[build_ply_plotly_fig] No .ply files found.")
        return None

    ply_path = ply_files[0]
    print(f"[build_ply_plotly_fig] Loading {ply_path}")

    try:
        from plyfile import PlyData

        ply   = PlyData.read(ply_path)
        verts = ply["vertex"]
        x = np.array(verts["x"], dtype=float)
        y = np.array(verts["y"], dtype=float)
        z = np.array(verts["z"], dtype=float)

        # ─ Vertex colours ─────────────────────────────────────────────────────────
        try:
            r_arr = np.array(verts["red"],   dtype=int)
            g_arr = np.array(verts["green"], dtype=int)
            b_arr = np.array(verts["blue"],  dtype=int)
            vertex_colors = [f"rgb({r_arr[n]},{g_arr[n]},{b_arr[n]})" for n in range(len(x))]
            has_colors = True
        except Exception:
            has_colors = False

        # ─ Faces ─────────────────────────────────────────────────────────────────
        i_idx, j_idx, k_idx = [], [], []
        if "face" in ply:
            for face in ply["face"]["vertex_indices"]:
                if len(face) == 3:
                    i_idx.append(int(face[0]))
                    j_idx.append(int(face[1]))
                    k_idx.append(int(face[2]))
        has_faces = bool(i_idx)

        # ─ Build Mesh3d trace ─────────────────────────────────────────────────────
        mesh_kwargs = dict(
            # remap axes: plot-x=x, plot-y=z, plot-z=-y  (upright figure)
            x=x, y=z, z=-y,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=200, z=150),
            hoverinfo="none",
            name=os.path.basename(ply_path),
        )
        if has_faces:
            mesh_kwargs["i"] = i_idx
            mesh_kwargs["j"] = j_idx
            mesh_kwargs["k"] = k_idx
        if has_colors:
            mesh_kwargs["vertexcolor"] = vertex_colors
        else:
            mesh_kwargs["color"]     = "#a0c4ff"
            mesh_kwargs["opacity"]   = 0.9

        trace = go.Mesh3d(**mesh_kwargs)

    except ImportError:
        print("[build_ply_plotly_fig] plyfile not installed — install with: pip install plyfile")
        return None
    except Exception as e:
        print(f"[build_ply_plotly_fig] Error reading PLY: {e}")
        return None

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=dict(text=f"🦴 3-D Mesh — {os.path.basename(ply_path)}",
                   font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="x", showbackground=False),
            yaxis=dict(title="z", showbackground=False),
            zaxis=dict(title="-y", showbackground=False),
            aspectmode="data",
            bgcolor="#f0f4f8",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        paper_bgcolor="#f0f4f8",
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 4. Master analysis function
# ─────────────────────────────────────────────────────────────────────────────
def run_full_lifting_analysis(image_path: str, prompt: str, openai_api_key: str) -> dict:
    """
    Orchestrate the full lifting analysis pipeline:
        1. Load SAM-3D keypoints from result JSON
        2. Compute pose metrics (F, A, D)
        3. Estimate M from Vision (gpt-4.1)
        4. Compute spinal loads
        5. Build 3-panel figure
        6. Return everything as a dict

    Returns dict with keys:
        BH, BW, M,
        flexion_deg, asymmetry_deg, reach_cm,
        spinal_loads (dict),
        plot_bytes (PNG bytes),
        error (str or None)
    """
    client = OpenAI(api_key=openai_api_key)

    # ── a. Body params from prompt ────────────────────────────────────────────
    body = extract_body_params_from_prompt(prompt, client)
    BH, BW = body["BH"], body["BW"]
    print(f"[analysis] BH={BH} cm  BW={BW} kg")

    # ── b. Load keypoints ─────────────────────────────────────────────────────
    result_dir = os.path.join(_ROOT, "sam 3d body result")
    json_path  = os.path.join(result_dir, "keypoints_3d.json")

    if not os.path.exists(json_path):
        return {"error": f"keypoints_3d.json not found at {json_path}. "
                         "Make sure sam3d_call() ran successfully."}

    with open(json_path, "r") as f:
        data = json.load(f)

    if "people" in data:
        kp_list = [p["keypoints_3d"] for p in data["people"] if "keypoints_3d" in p]
    elif "keypoints_3d" in data:
        kp_list = [data["keypoints_3d"]]
    else:
        kp_list = [data]

    if not kp_list:
        return {"error": "No keypoints found in keypoints_3d.json"}

    pts = np.array(kp_list[0])   # use first detected person

    # ── c. Pose metrics ───────────────────────────────────────────────────────
    F, _, _, _   = compute_flexion(pts)
    A, _, _      = compute_asymmetry(pts)
    D, scale, _, _, _ = compute_reach_cm(pts, BH)

    print(f"[analysis] F={F:.2f}°  A={A:.2f}°  D={D:.2f} cm")

    # ── d. Object weight (M) via Vision ───────────────────────────────────────
    M = estimate_object_weight(image_path, client)
    print(f"[analysis] M={M} kg")

    # ── e. Spinal loads ───────────────────────────────────────────────────────
    sex = 0   # always male, as specified
    if F < 20.0:
        loads = compute_spinal_loads_standing(sex=sex, BH=BH, BW=BW, M=M, A=A, D=D)
        model_used = "standing (F < 20°)"
    else:
        loads = compute_spinal_loads_flex(sex=sex, BH=BH, BW=BW, M=M, A=A, F=F, D=D)
        model_used = "flexion (F ≥ 20°)"

    print(f"[analysis] Spinal loads ({model_used}): {loads}")

    # ── f. Two-panel static figure ───────────────────────────────────────────────
    try:
        plot_bytes = _plot_two_panel(pts, image_path)
    except Exception as e:
        print(f"[analysis] Plot error (non-fatal): {e}")
        plot_bytes = None

    # ── g. Interactive PLY Mesh3d figure ─────────────────────────────────────────
    ply_fig = build_ply_plotly_fig(result_dir)

    return {
        "BH": BH, "BW": BW, "M": M,
        "flexion_deg":       round(F, 2),
        "asymmetry_deg":     round(A, 2),
        "reach_cm":          round(D, 2),
        "spinal_loads":      {k: round(v, 1) for k, v in loads.items()},
        "model_used":        model_used,
        "scale_cm_per_unit": round(scale, 5),
        "plot_bytes":        plot_bytes,
        "ply_fig":           ply_fig,    # plotly Figure or None
        "error":             None,
    }
