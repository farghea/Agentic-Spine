"""
utils_sensitivity.py
--------------------
Sensitivity Analysis / FF-Mode (Full Flexibility Mode) pipeline for the
OpenSim Spine Musculoskeletal Agent.

Trigger keywords
----------------
Any of:  "sensitivity analysis", "sensitivity", "ff-mode",
         "full flexibility mode", "parametric study/analysis",
         "perturbation analysis/study"

When triggered, a dedicated agentic branch is activated instead of the
normal OpenSim pipeline.  The existing pipeline nodes are NOT called
(except for model selection, reused internally by the planner).

5 Built-in Perturbation Tools
------------------------------
1. flexion_sweep          — trunk-flexion angle ± delta using the existing
                            interpolation library (0°–90° clamped)
2. slack_length           — scales ALL tendon_slack_length values by ± pct%
3. vertebral_translation  — shifts ALL IVDjnt location_in_parent X by ± mm
4. gravity                — scales <gravity> Y component by ± pct%
5. combinatorial          — all 2^n combinations of selected tools (±) + 1 baseline

Tools 2-4 are implemented in utils_full_functional_model.py (XML-based,
no OpenSim API needed in the main conda environment).

Output DataFrames
-----------------
Three extended DataFrames stored in state["dataframes"]:
  "spinal"      — IVD joint loads (wide: force_fx / force_fy / force_fz)
  "forces"      — muscle forces (N)
  "activations" — muscle activations (0–1)

All three include:
  Perturbation_Type   str    e.g. "flexion_angle", "slack_length_pct"
  Perturbation_Value  float  e.g. 15.0, 5.0, -5.0

Agentic Executor Loop
---------------------
The executor node loops over all cases and uses up to MAX_REFLECTIONS (10)
LLM-guided reflection steps when simulations fail.  At each reflection the
LLM chooses one of:  clamp_angle | skip | use_default_activity | retry.

Programmatic repairs (e.g. angle clamping) are attempted before calling
the LLM to save API calls.

LangGraph Nodes exported
------------------------
  sensitivity_planner_node
  sensitivity_executor_node
  sensitivity_analyst_node
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sys
from itertools import product as _itertools_product
from typing import Optional

import numpy as np
import pandas as pd
import google.generativeai as genai
from openai import OpenAI

# ── Interpolation generators (same env, no OpenSim) ──────────────────────────
_INTERP_DIR = os.path.join(
    os.path.dirname(__file__), "opensim_files", "motion force interpolation"
)
if _INTERP_DIR not in sys.path:
    sys.path.insert(0, _INTERP_DIR)

from interpolate_mot import generate_interpolated_mot
from generate_force_mot import generate_force_mot

# ── XML model modifiers ───────────────────────────────────────────────────────
from utils_full_functional_model import (
    copy_osim_to_temp,
    validate_osim_xml,
    get_osim_stats,
    apply_slack_length_perturbation,
    apply_vertebral_translation,
    apply_gravity_scale,
    apply_combined_perturbation,
)

# ── Existing simulation runner + model constants ──────────────────────────────
from utils import run_opensim_simulation, MODEL, MODEL_TYPE

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

SENSITIVITY_KEYWORDS = [
    "sensitivity analysis",
    "sensitivity study",
    "sensitivity",
    "ff-mode",
    "full flexibility mode",
    "ffmode",
    "parametric study",
    "parametric analysis",
    "perturbation analysis",
    "perturbation study",
]

# Defaults used when the user does not specify values
DEFAULT_FLEXION_DELTA   = 5.0    # degrees
DEFAULT_SLACK_PCT       = 5.0    # percent
DEFAULT_VERTEBRAL_MM    = 2.0    # millimetres
DEFAULT_GRAVITY_PCT     = 10.0   # percent

FLEXION_MIN = 0.0                # interpolation library hard limit
FLEXION_MAX = 90.0

DEFAULT_ACTIVITY_ID = 2          # "Neutral standing" (used as fallback baseline)
TEMP_DIR_NAME       = "sensitivity_temp"  # sub-folder inside opensim_files/
MAX_REFLECTIONS     = 10


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KEYWORD DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def is_sensitivity_prompt(text: str) -> bool:
    """Return True if *text* contains any FF-mode / sensitivity keyword."""
    t = text.lower()
    return any(kw in t for kw in SENSITIVITY_KEYWORDS)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOOL FUNCTIONS (executor uses these to avoid getting stuck)
# ════════════════════════════════════════════════════════════════════════════════

def tool_clamp_flexion_angle(angle: float) -> float:
    """Clamp *angle* to the valid interpolation range [0°, 90°]."""
    clamped = max(FLEXION_MIN, min(FLEXION_MAX, float(angle)))
    if clamped != float(angle):
        print(f"  [TOOL clamp_angle] {angle}° → {clamped}°")
    return clamped


def tool_get_base_motion_force_paths(activity_id: int = DEFAULT_ACTIVITY_ID) -> dict:
    """
    Return the *relative* motion and force file paths for a standard activity.
    run_opensim_simulation() resolves relative paths automatically.
    """
    return {
        "motion_path": f"NMB_Motion{activity_id}.mot",
        "force_path":  f"NMB_ExternalForce{activity_id}.mot",
    }


def tool_validate_case(case: dict) -> tuple:
    """
    Check that a case dict has all required keys and that absolute-path files
    actually exist on disk.

    Returns (True, "") or (False, error_message).
    """
    required = [
        "label", "model_path", "motion_path", "force_path",
        "perturbation_type", "perturbation_value",
    ]
    for key in required:
        if key not in case:
            return False, f"Missing required field: '{key}'"

    for field in ("model_path", "motion_path", "force_path"):
        p = case[field]
        if os.path.isabs(p) and not os.path.exists(p):
            return False, f"File not found: {p}"

    return True, ""


def tool_skip_case(case: dict, reason: str) -> dict:
    """Return a 'Skipped' result dict for *case*."""
    return {
        "label":              case.get("label", "unknown"),
        "model":              os.path.basename(case.get("model_path", "")),
        "perturbation_type":  case.get("perturbation_type", ""),
        "perturbation_value": case.get("perturbation_value", None),
        "activity_label":     case.get("activity_label", ""),
        "status":             "Skipped",
        "error":              reason,
        "spinal_loads":       None,
        "muscle_forces":      None,
        "muscle_activations": None,
    }


def tool_list_case_statuses(cases: list, results: list) -> dict:
    """Return a human-readable status summary dict."""
    done    = {r["label"] for r in results}
    pending = [c for c in cases if c["label"] not in done]
    return {
        "total":           len(cases),
        "completed":       len(done),
        "pending":         len(pending),
        "success":         sum(1 for r in results if r["status"] == "Success"),
        "failed":          sum(1 for r in results if r["status"] == "Failed"),
        "skipped":         sum(1 for r in results if r["status"] == "Skipped"),
        "pending_labels":  [c["label"] for c in pending[:5]],
    }


def tool_get_model_stats(model_path: str) -> dict:
    """Verify a (possibly modified) .osim file by returning key stats."""
    return get_osim_stats(model_path)


def tool_validate_osim_xml(model_path: str) -> dict:
    """Check that a modified .osim file is still parseable XML."""
    valid, msg = validate_osim_xml(model_path)
    return {"valid": valid, "message": msg or "OK", "path": model_path}


def tool_repair_flexion_case(case: dict, error_msg: str, output_dir: str) -> Optional[dict]:
    """
    Attempt a programmatic repair for a failed flexion case.

    Strategy: clamp the angle to [0°, 90°] and regenerate the .mot files.
    Returns a repaired case dict, or None if the case cannot be repaired here.
    """
    if case.get("perturbation_type") != "flexion_angle":
        return None

    angle = case.get("perturbation_value")
    if angle is None:
        return None

    clamped = tool_clamp_flexion_angle(float(angle))
    if abs(clamped - float(angle)) < 1e-6:
        # Angle was already in range → not an angle-clamping issue
        return None

    try:
        wph    = float(case.get("weight_per_hand_kg", 0.0))
        load_n = wph * -9.81 if wph else 0.0

        motion_files = generate_interpolated_mot([clamped], output_dir=output_dir)
        force_files  = generate_force_mot(load_n=load_n, output_dir=output_dir)

        repaired = dict(case)
        repaired["motion_path"]        = motion_files[0]
        repaired["force_path"]         = force_files[0]
        repaired["perturbation_value"] = clamped
        repaired["label"]              = case["label"] + f"_clamped{int(clamped)}"
        repaired["activity_label"]     = f"{clamped}° trunk flexion (clamped)"
        print(f"  [TOOL repair_flexion] Rebuilt case at {clamped}°")
        return repaired
    except Exception as exc:
        print(f"  [TOOL repair_flexion] Could not repair: {exc}")
        return None


def tool_use_default_activity(case: dict) -> dict:
    """
    Replace a case's motion/force paths with the neutral-standing defaults.
    Useful when a custom interpolated .mot file is missing or corrupt.
    """
    paths    = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
    repaired = dict(case)
    repaired["motion_path"]    = paths["motion_path"]
    repaired["force_path"]     = paths["force_path"]
    repaired["activity_label"] = "Neutral standing (fallback)"
    repaired["label"]          = case.get("label", "case") + "_fallback"
    return repaired


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FLEXION VARIANT GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def generate_flexion_variants(
    base_angle:         float,
    delta_deg:          float,
    weight_per_hand_kg: float,
    output_dir:         str,
    model_path:         str,
    model_filename:     str,
) -> list:
    """
    Generate base−delta, base, base+delta trunk-flexion simulation cases.

    One shared force file is created (same load for all angles).
    Angles outside [0°, 90°] are clamped and a warning is printed.

    Returns a list of case dicts.
    """
    angles_raw = [
        base_angle - delta_deg,
        base_angle,
        base_angle + delta_deg,
    ]
    # Clamp to valid range and deduplicate
    angles = sorted(set(tool_clamp_flexion_angle(a) for a in angles_raw))

    load_n = float(weight_per_hand_kg) * -9.81 if weight_per_hand_kg else 0.0

    try:
        motion_files = generate_interpolated_mot(angles, output_dir=output_dir)
    except Exception as exc:
        raise RuntimeError(f"generate_interpolated_mot failed: {exc}") from exc

    try:
        force_files = generate_force_mot(load_n=load_n, output_dir=output_dir)
        force_path  = force_files[0]  # same load for all angles
    except Exception as exc:
        raise RuntimeError(f"generate_force_mot failed: {exc}") from exc

    cases = []
    for i, angle in enumerate(angles):
        load_tag = f"_load{weight_per_hand_kg:.1f}kg" if weight_per_hand_kg else ""
        label    = f"flex_{int(angle)}deg{load_tag}_{model_filename}"
        cases.append({
            "label":               label,
            "model_path":          model_path,
            "motion_path":         motion_files[i],
            "force_path":          force_path,
            "perturbation_type":   "flexion_angle",
            "perturbation_value":  angle,
            "activity_label":      f"{angle}° trunk flexion",
            "weight_per_hand_kg":  weight_per_hand_kg,
        })

    print(f"  [flexion_sweep] Generated {len(cases)} cases for base={base_angle}° ±{delta_deg}°")
    return cases


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL-MODIFICATION CASE BUILDERS (Tools 2, 3, 4)
# ════════════════════════════════════════════════════════════════════════════════

def _build_slack_cases(
    plan:            dict,
    model_path:      str,
    model_filename:  str,
    temp_dir:        str,
    activity_motion: str,
    activity_force:  str,
    activity_label:  str,
) -> list:
    """Build base + ±pct slack-length cases (3 modified .osim files)."""
    pct      = float(plan.get("slack_length_pct", DEFAULT_SLACK_PCT))
    cases    = []
    base_fn  = os.path.splitext(model_filename)[0]

    variants = [
        (1.0,             "base",                   0.0),
        (1.0 + pct / 100, f"slack_plus{int(pct)}pct",  pct),
        (1.0 - pct / 100, f"slack_minus{int(pct)}pct", -pct),
    ]

    for scale, tag, pct_val in variants:
        out_path = os.path.join(temp_dir, f"{base_fn}_{tag}.osim")

        if abs(scale - 1.0) < 1e-9:
            shutil.copy2(model_path, out_path)
        else:
            try:
                apply_slack_length_perturbation(model_path, scale, out_path)
            except Exception as exc:
                print(f"  [slack_cases] Failed to create {tag}: {exc} — skipping")
                continue

        valid, err = validate_osim_xml(out_path)
        if not valid:
            print(f"  [slack_cases] XML invalid after modification ({tag}): {err} — skipping")
            continue

        cases.append({
            "label":              f"slack_{tag}_{model_filename}",
            "model_path":         out_path,
            "motion_path":        activity_motion,
            "force_path":         activity_force,
            "perturbation_type":  "slack_length_pct",
            "perturbation_value": pct_val,
            "activity_label":     activity_label,
        })

    print(f"  [slack_length] Built {len(cases)} cases (±{pct}%)")
    return cases


def _build_vertebral_cases(
    plan:            dict,
    model_path:      str,
    model_filename:  str,
    temp_dir:        str,
    activity_motion: str,
    activity_force:  str,
    activity_label:  str,
) -> list:
    """Build base + ±mm vertebral-translation cases (3 modified .osim files)."""
    delta_mm = float(plan.get("vertebral_delta_mm", DEFAULT_VERTEBRAL_MM))
    delta_m  = delta_mm / 1000.0
    cases    = []
    base_fn  = os.path.splitext(model_filename)[0]

    variants = [
        (0.0,      "base",                     0.0),
        (delta_m,  f"vert_plus{delta_mm:.0f}mm",   delta_mm),
        (-delta_m, f"vert_minus{delta_mm:.0f}mm", -delta_mm),
    ]

    for dx, tag, mm_val in variants:
        out_path = os.path.join(temp_dir, f"{base_fn}_{tag}.osim")

        if abs(dx) < 1e-12:
            shutil.copy2(model_path, out_path)
        else:
            try:
                apply_vertebral_translation(model_path, dx, out_path)
            except Exception as exc:
                print(f"  [vert_cases] Failed to create {tag}: {exc} — skipping")
                continue

        valid, err = validate_osim_xml(out_path)
        if not valid:
            print(f"  [vert_cases] XML invalid after modification ({tag}): {err} — skipping")
            continue

        cases.append({
            "label":              f"vert_{tag}_{model_filename}",
            "model_path":         out_path,
            "motion_path":        activity_motion,
            "force_path":         activity_force,
            "perturbation_type":  "vertebral_x_mm",
            "perturbation_value": mm_val,
            "activity_label":     activity_label,
        })

    print(f"  [vertebral_translation] Built {len(cases)} cases (±{delta_mm} mm)")
    return cases


def _build_gravity_cases(
    plan:            dict,
    model_path:      str,
    model_filename:  str,
    temp_dir:        str,
    activity_motion: str,
    activity_force:  str,
    activity_label:  str,
) -> list:
    """Build base + ±pct gravity cases (3 modified .osim files)."""
    pct     = float(plan.get("gravity_pct", DEFAULT_GRAVITY_PCT))
    cases   = []
    base_fn = os.path.splitext(model_filename)[0]

    variants = [
        (1.0,             "base",                  0.0),
        (1.0 + pct / 100, f"grav_plus{int(pct)}pct",  pct),
        (1.0 - pct / 100, f"grav_minus{int(pct)}pct", -pct),
    ]

    for gscale, tag, pct_val in variants:
        out_path = os.path.join(temp_dir, f"{base_fn}_{tag}.osim")

        if abs(gscale - 1.0) < 1e-9:
            shutil.copy2(model_path, out_path)
        else:
            try:
                apply_gravity_scale(model_path, gscale, out_path)
            except Exception as exc:
                print(f"  [grav_cases] Failed to create {tag}: {exc} — skipping")
                continue

        valid, err = validate_osim_xml(out_path)
        if not valid:
            print(f"  [grav_cases] XML invalid after modification ({tag}): {err} — skipping")
            continue

        cases.append({
            "label":              f"grav_{tag}_{model_filename}",
            "model_path":         out_path,
            "motion_path":        activity_motion,
            "force_path":         activity_force,
            "perturbation_type":  "gravity_pct",
            "perturbation_value": pct_val,
            "activity_label":     activity_label,
        })

    print(f"  [gravity] Built {len(cases)} cases (±{pct}%)")
    return cases


def _build_combinatorial_cases(
    plan:            dict,
    model_path:      str,
    model_filename:  str,
    temp_dir:        str,
) -> list:
    """
    Build 1 baseline + 2^n combination cases for the given combo_tools.

    Each dimension contributes two sign-variants (+/-).  Every unique
    combination of signs across all dimensions becomes one simulation case.

    Supported combo dimensions (specified in plan under "combo_tools"):
      - "flexion_sweep"        : motion file angle = base ± delta_deg
      - "slack_length"         : .osim tendon_slack_length scaled ± pct%
      - "vertebral_translation": .osim IVDjnt X-location shifted ± mm

    Plan keys consumed
    ------------------
    combo_tools           : list[str]  which dimensions to combine
    combo_flexion_base    : float  base flexion angle (deg, default 20)
    combo_flexion_delta   : float  ± delta (deg, default 5)
    combo_slack_pct       : float  slack perturbation % (default 5)
    combo_vertebral_mm    : float  vertebral perturbation mm (default 2)
    combo_weight_per_hand : float  hand load kg (default 0)
    """
    combo_tools  = plan.get("combo_tools",
                            ["flexion_sweep", "slack_length", "vertebral_translation"])
    base_angle   = float(plan.get("combo_flexion_base",    20.0))
    delta_deg    = float(plan.get("combo_flexion_delta",    5.0))
    slack_pct    = float(plan.get("combo_slack_pct",        5.0))
    vert_mm      = float(plan.get("combo_vertebral_mm",     2.0))
    wph          = float(plan.get("combo_weight_per_hand",  0.0))

    load_n  = wph * -9.81 if wph else 0.0
    base_fn = os.path.splitext(model_filename)[0]

    # ── Dimension map:  name -> list of (tag_label, sign_value) ──────────────
    all_dims = {
        "flexion_sweep":         [("minus", base_angle - delta_deg),
                                  ("plus",  base_angle + delta_deg)],
        "slack_length":          [("minus", -slack_pct),
                                  ("plus",  slack_pct)],
        "vertebral_translation": [("minus", -vert_mm),
                                  ("plus",  vert_mm)],
    }

    active_dims = [(dim, all_dims[dim])
                   for dim in combo_tools if dim in all_dims]

    if not active_dims:
        print("  [combinatorial] No recognised combo_tools — returning 0 cases")
        return []

    dim_names  = [d[0] for d in active_dims]
    dim_values = [d[1] for d in active_dims]

    # ── Baseline motion/force (clamped base angle) ────────────────────────────
    base_clamped = tool_clamp_flexion_angle(base_angle)
    try:
        base_motion = generate_interpolated_mot([base_clamped], output_dir=temp_dir)[0]
        base_force  = generate_force_mot(load_n=load_n,          output_dir=temp_dir)[0]
    except Exception as exc:
        print(f"  [combinatorial] Baseline motion error ({exc}) — using neutral standing")
        _paths     = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
        base_motion = _paths["motion_path"]
        base_force  = _paths["force_path"]

    cases = []

    # ── 1. Baseline case (no perturbation) ────────────────────────────────────
    cases.append({
        "label":              f"combo_baseline_{model_filename}",
        "model_path":         model_path,
        "motion_path":        base_motion,
        "force_path":         base_force,
        "perturbation_type":  "combinatorial",
        "perturbation_value": 0.0,
        "activity_label":     f"{base_clamped}deg flexion (combo baseline)",
        "combo_details":      {d: 0.0 for d in dim_names},
    })

    # ── 2. All 2^n combinations ────────────────────────────────────────────────
    for combo in _itertools_product(*dim_values):
        # combo: tuple of (sign_label, value) pairs, one per active dimension
        label_parts  = []
        combo_vals   = {}
        flexion_angle = base_clamped
        slack_scale   = 1.0
        vert_dx_m     = 0.0

        for i, (sign_label, value) in enumerate(combo):
            dim = dim_names[i]
            combo_vals[dim] = float(value)
            label_parts.append(f"{dim[:4]}_{sign_label}")

            if dim == "flexion_sweep":
                flexion_angle = tool_clamp_flexion_angle(float(value))
            elif dim == "slack_length":
                slack_scale = 1.0 + float(value) / 100.0
            elif dim == "vertebral_translation":
                vert_dx_m = float(value) / 1000.0   # mm → m

        combo_label = "_".join(label_parts)

        # Generate motion/force for this flexion angle
        try:
            motion_files = generate_interpolated_mot([flexion_angle], output_dir=temp_dir)
            force_files  = generate_force_mot(load_n=load_n,           output_dir=temp_dir)
            motion_path  = motion_files[0]
            force_path   = force_files[0]
        except Exception as exc:
            print(f"  [combinatorial] Motion error for {combo_label}: {exc} — using baseline")
            motion_path = base_motion
            force_path  = base_force

        # Build combined modified model
        out_model = os.path.join(temp_dir, f"{base_fn}_combo_{combo_label}.osim")
        try:
            apply_combined_perturbation(
                osim_path=model_path,
                output_path=out_model,
                slack_scale=slack_scale,
                vertebral_delta_m=vert_dx_m,
                gravity_scale=1.0,
            )
            valid, err = validate_osim_xml(out_model)
            if not valid:
                print(f"  [combinatorial] Invalid XML for {combo_label}: {err} — skipping")
                continue
        except Exception as exc:
            print(f"  [combinatorial] Model error for {combo_label}: {exc} — skipping")
            continue

        cases.append({
            "label":              f"combo_{combo_label}_{model_filename}",
            "model_path":         out_model,
            "motion_path":        motion_path,
            "force_path":         force_path,
            "perturbation_type":  "combinatorial",
            "perturbation_value": combo_vals,
            "activity_label":     (f"{flexion_angle}deg flexion "
                                   f"slack={slack_scale:.3f}x "
                                   f"vert={vert_dx_m*1000:.1f}mm"),
            "combo_details":      combo_vals,
        })

    n_combos = len(cases) - 1   # subtract baseline
    print(f"  [combinatorial] Built {len(cases)} cases "
          f"(1 baseline + {n_combos} of 2^{len(dim_names)}={2**len(dim_names)})")
    return cases


def _build_custom_cases(
    tool:            str,
    plan:            dict,
    model_path:      str,
    model_filename:  str,
    temp_dir:        str,
    activity_motion: str,
    activity_force:  str,
    activity_label:  str,
) -> list:
    """
    Dynamically generate python code to modify the .osim file for a custom tool,
    and execute it to build base + ±pct custom cases.
    """
    user_prompt = plan.get("user_prompt", "")
    custom_vals = plan.get("custom_tool_values", {})
    pct         = float(custom_vals.get(tool, 5.0) if custom_vals else 5.0)
    cases       = []
    base_fn     = os.path.splitext(model_filename)[0]

    # Generate custom modification function using LLM
    print(f"  [custom_tool] Generating code for custom tool '{tool}'...")
    try:
        keys = _load_keys()
    except Exception as exc:
        raise RuntimeError(f"Cannot load keys: {exc}")

    prompt = f"""
You are writing a surgical Python 3 function to modify an OpenSim `.osim` model file (XML format) for a sensitivity analysis.
Do NOT use `xml.etree.ElementTree` because it reformats the entire XML and breaks the OpenSim parser. Instead, use Python regular expressions (`re`) or simple string replacements on the file text.

CUSTOM TOOL NAME: "{tool}"
USER PROMPT: "{user_prompt}"

TASK:
Write a Python function with the exact signature:
`def modify_model(input_osim: str, value: float, output_osim: str) -> None`

Where:
- `input_osim`: path to the source .osim file.
- `value`: the scale factor or delta to apply (e.g. 1.05 for +5%, 0.95 for -5%).
- `output_osim`: path where the modified copy should be written.

KNOWLEDGE:
- Analyze the custom tool name and the user prompt to identify which XML tag(s) in the OpenSim `.osim` model file represent the target parameter.
- Use Python regular expressions (`re`) to surgically search and scale/offset those XML tags by `value`.
- Example pattern for scaling a generic tag like `<optimal_fiber_length>` (adapt this to target the correct tag for your custom tool!):
  ```python
  import re
  def modify_model(input_osim, value, output_osim):
      with open(input_osim, 'r', encoding='utf-8') as f:
          content = f.read()
      # Match `<tag>val</tag>` and scale the float value
      modified = re.sub(
          r'<optimal_fiber_length>([\\d.eE+\\-]+)</optimal_fiber_length>',
          lambda m: f'<optimal_fiber_length>{{float(m.group(1)) * value:.10f}}</optimal_fiber_length>',
          content
      )
      with open(output_osim, 'w', encoding='utf-8') as f:
          f.write(modified)
  ```

Return ONLY the Python code block starting with `import re` etc. Do NOT include markdown code blocks (e.g. no ```python) or any other conversational text. Output the raw Python text.
"""

    code_content = ""
    if MODEL == "openai":
        client   = OpenAI(api_key=keys["openai_api_key"])
        response = client.chat.completions.create(
            model=MODEL_TYPE,
            messages=[
                {"role": "system", "content": "Return only raw Python code. No markdown formatting, no explanations."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
        )
        code_content = response.choices[0].message.content
    elif MODEL == "gemini":
        genai.configure(api_key=keys["gemini_api_key"])
        gem   = genai.GenerativeModel(MODEL_TYPE)
        resp  = gem.generate_content(prompt)
        code_content = resp.text
    else:
        raise RuntimeError(f"Unknown MODEL: {MODEL}")

    # Strip any markdown formatting in case the model ignored system prompts
    code_content = code_content.replace("```python", "").replace("```", "").strip()

    # Save to a temporary file
    code_filename = f"custom_mod_{tool}_{datetime.datetime.now().strftime('%H%M%S')}.py"
    code_path     = os.path.join(temp_dir, code_filename)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code_content)

    # Dynamically load the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_tool_mod", code_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    modify_func = mod.modify_model

    variants = [
        (1.0,             "base",                  0.0),
        (1.0 + pct / 100, f"{tool}_plus{int(pct)}pct",  pct),
        (1.0 - pct / 100, f"{tool}_minus{int(pct)}pct", -pct),
    ]

    for scale, tag, pct_val in variants:
        out_path = os.path.join(temp_dir, f"{base_fn}_{tag}.osim")

        if abs(scale - 1.0) < 1e-9:
            shutil.copy2(model_path, out_path)
        else:
            try:
                modify_func(model_path, scale, out_path)
            except Exception as exc:
                print(f"  [custom_tool] Failed to apply {tag}: {exc} — skipping")
                continue

        valid, err = validate_osim_xml(out_path)
        if not valid:
            print(f"  [custom_tool] XML invalid after modification ({tag}): {err} — skipping")
            continue

        cases.append({
            "label":              f"{tool}_{tag}_{model_filename}",
            "model_path":         out_path,
            "motion_path":        activity_motion,
            "force_path":         activity_force,
            "perturbation_type":  tool,
            "perturbation_value": pct_val,
            "activity_label":     activity_label,
        })

    print(f"  [custom_tool] Built {len(cases)} cases for '{tool}' (±{pct}%)")
    return cases


# Known tool names — anything outside this set triggers a graceful warning
_KNOWN_TOOLS = frozenset([
    "flexion_sweep", "slack_length", "vertebral_translation",
    "gravity", "combinatorial",
])


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MASTER CASE BUILDER
# ════════════════════════════════════════════════════════════════════════════════

def _build_all_sensitivity_cases(plan: dict, selected_models: list) -> list:
    """
    Build the complete list of simulation cases for all requested tools
    and all selected models.

    All generated files (modified .osim, interpolated .mot) are placed in
    a timestamped temp directory: ``opensim_files/sensitivity_temp/<run_id>/``

    Returns a flat list of case dicts.
    """
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.abspath(
        os.path.join(os.getcwd(), "opensim_files", TEMP_DIR_NAME, run_id)
    )
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\n  [sensitivity] Temp dir: {temp_dir}")

    tools              = plan.get("tools", ["flexion_sweep", "slack_length", "vertebral_translation", "gravity"])
    flexion_activities = plan.get("flexion_activities", [])

    # ── Determine baseline motion/force for model-modification tools ──────────
    # Use the first flexion activity's angle if specified; else neutral standing.
    if flexion_activities:
        fa           = flexion_activities[0]
        base_angle   = tool_clamp_flexion_angle(float(fa.get("base_angle", 0)))
        base_wph     = float(fa.get("weight_per_hand_kg", 0.0))
        base_load_n  = base_wph * -9.81 if base_wph else 0.0
        try:
            base_motion = generate_interpolated_mot([base_angle], output_dir=temp_dir)[0]
            base_force  = generate_force_mot(load_n=base_load_n, output_dir=temp_dir)[0]
            base_act_lbl = f"{base_angle}° trunk flexion"
        except Exception as exc:
            print(f"  [builder] Warning: could not generate baseline flexion files ({exc}). Using neutral standing.")
            paths        = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
            base_motion  = paths["motion_path"]
            base_force   = paths["force_path"]
            base_act_lbl = "Neutral standing"
    else:
        paths        = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
        base_motion  = paths["motion_path"]
        base_force   = paths["force_path"]
        base_act_lbl = "Neutral standing"

    all_cases = []

    # Directory of the opensim_files folder (sibling of this script)
    opensim_files_dir = os.path.join(os.getcwd(), "opensim_files")

    for model_info in selected_models:
        model_path     = model_info.get("full_path", "")
        model_filename = model_info.get("Filename", os.path.basename(model_path))

        if not model_path:
            print(f"  [builder] Empty model path — skipping")
            continue

        # full_path from model_selection_node is relative to opensim_files/
        # (e.g. "Male/Age6069/599_...osim").
        # Resolve to absolute so os.path.exists() works correctly.
        # This mirrors exactly what simulation_node does before calling run_opensim_simulation.
        if not os.path.isabs(model_path):
            model_path = os.path.join(opensim_files_dir, model_path)

        if not os.path.exists(model_path):
            print(f"  [builder] Model not found: {model_path} — skipping")
            continue

        # ── Tool 1: Flexion sweep ─────────────────────────────────────────────
        if "flexion_sweep" in tools:
            if flexion_activities:
                for fa in flexion_activities:
                    angle  = float(fa.get("base_angle", 0))
                    delta  = float(fa.get("delta_deg", DEFAULT_FLEXION_DELTA))
                    wph    = float(fa.get("weight_per_hand_kg", 0.0))
                    try:
                        cases = generate_flexion_variants(
                            base_angle=angle,
                            delta_deg=delta,
                            weight_per_hand_kg=wph,
                            output_dir=temp_dir,
                            model_path=model_path,
                            model_filename=model_filename,
                        )
                        all_cases.extend(cases)
                    except Exception as exc:
                        print(f"  [builder] flexion_sweep error: {exc}")
            else:
                # Default: 0° ± delta
                try:
                    cases = generate_flexion_variants(
                        base_angle=0.0,
                        delta_deg=DEFAULT_FLEXION_DELTA,
                        weight_per_hand_kg=0.0,
                        output_dir=temp_dir,
                        model_path=model_path,
                        model_filename=model_filename,
                    )
                    all_cases.extend(cases)
                except Exception as exc:
                    print(f"  [builder] default flexion_sweep error: {exc}")

        # ── Tool 2: Slack length ─────────────────────────────────────────────
        if "slack_length" in tools:
            try:
                cases = _build_slack_cases(
                    plan=plan,
                    model_path=model_path,
                    model_filename=model_filename,
                    temp_dir=temp_dir,
                    activity_motion=base_motion,
                    activity_force=base_force,
                    activity_label=base_act_lbl,
                )
                all_cases.extend(cases)
            except Exception as exc:
                print(f"  [builder] slack_length error: {exc}")

        # ── Tool 3: Vertebral translation ────────────────────────────────────
        if "vertebral_translation" in tools:
            try:
                cases = _build_vertebral_cases(
                    plan=plan,
                    model_path=model_path,
                    model_filename=model_filename,
                    temp_dir=temp_dir,
                    activity_motion=base_motion,
                    activity_force=base_force,
                    activity_label=base_act_lbl,
                )
                all_cases.extend(cases)
            except Exception as exc:
                print(f"  [builder] vertebral_translation error: {exc}")

        # ── Tool 4: Gravity ──────────────────────────────────────────────────
        if "gravity" in tools:
            try:
                cases = _build_gravity_cases(
                    plan=plan,
                    model_path=model_path,
                    model_filename=model_filename,
                    temp_dir=temp_dir,
                    activity_motion=base_motion,
                    activity_force=base_force,
                    activity_label=base_act_lbl,
                )
                all_cases.extend(cases)
            except Exception as exc:
                print(f"  [builder] gravity error: {exc}")

        # ── Tool 5: Combinatorial (2^n combinations) ─────────────────────────
        if "combinatorial" in tools:
            try:
                cases = _build_combinatorial_cases(
                    plan=plan,
                    model_path=model_path,
                    model_filename=model_filename,
                    temp_dir=temp_dir,
                )
                all_cases.extend(cases)
            except Exception as exc:
                print(f"  [builder] combinatorial error: {exc}")

        # ── Unknown/Custom tools → dynamic code generation ──────────────────
        for tool in tools:
            if tool not in _KNOWN_TOOLS:
                try:
                    custom_cases = _build_custom_cases(
                        tool=tool,
                        plan=plan,
                        model_path=model_path,
                        model_filename=model_filename,
                        temp_dir=temp_dir,
                        activity_motion=base_motion,
                        activity_force=base_force,
                        activity_label=base_act_lbl,
                    )
                    all_cases.extend(custom_cases)
                except Exception as exc:
                    print(f"  [builder] Dynamic custom tool '{tool}' failed: {exc}")

    print(f"\n  [sensitivity] Total cases built: {len(all_cases)}")
    return all_cases


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SIMULATION RUNNER
# ════════════════════════════════════════════════════════════════════════════════

def _run_single_sensitivity_case(case: dict) -> dict:
    """
    Run one sensitivity simulation case using the existing OpenSim subprocess.

    Validates the case first (file existence, required fields), then delegates
    to run_opensim_simulation() from utils.py.

    Returns a result dict always containing:
      label, model, perturbation_type, perturbation_value, activity_label,
      status ("Success" | "Failed" | "Skipped"), error,
      spinal_loads, muscle_forces, muscle_activations
    """
    label = case.get("label", "unknown")
    print(f"  [run] {label}")

    # 1. Validate before running
    valid, err = tool_validate_case(case)
    if not valid:
        print(f"    → VALIDATION FAILED: {err}")
        return _failed_result(case, err)

    # 2. Run
    try:
        spinal_loads, muscle_forces, muscle_activations = run_opensim_simulation(
            case["model_path"],
            case["motion_path"],
            case["force_path"],
        )

        if spinal_loads is None and muscle_forces is None:
            raise RuntimeError("Simulation returned None — likely a subprocess crash.")

        return {
            "label":              label,
            "model":              os.path.basename(case["model_path"]),
            "perturbation_type":  case.get("perturbation_type", ""),
            "perturbation_value": case.get("perturbation_value", None),
            "activity_label":     case.get("activity_label", ""),
            "status":             "Success",
            "error":              None,
            "spinal_loads":       spinal_loads,
            "muscle_forces":      muscle_forces,
            "muscle_activations": muscle_activations,
        }

    except Exception as exc:
        msg = str(exc)
        print(f"    → FAILED: {msg[:120]}")
        return _failed_result(case, msg)


def _failed_result(case: dict, error: str) -> dict:
    return {
        "label":              case.get("label", "unknown"),
        "model":              os.path.basename(case.get("model_path", "")),
        "perturbation_type":  case.get("perturbation_type", ""),
        "perturbation_value": case.get("perturbation_value", None),
        "activity_label":     case.get("activity_label", ""),
        "status":             "Failed",
        "error":              error,
        "spinal_loads":       None,
        "muscle_forces":      None,
        "muscle_activations": None,
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LLM REFLECTION ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def _load_keys() -> dict:
    with open("info_and_keys.json") as f:
        return json.load(f)


def _llm_reflect_on_error(
    case:       dict,
    error_msg:  str,
    plan:       dict,
    iteration:  int,
    output_dir: str,
) -> Optional[dict]:
    """
    Use the LLM to suggest a repair action for a failed case.

    Tries programmatic fixes first (angle clamping) before calling the LLM.
    Returns a repaired case dict, or None to skip the case.

    Possible LLM actions:
      clamp_angle          → rebuild flexion .mot files with clamped angle
      use_default_activity → swap motion/force to neutral standing
      retry                → retry the exact same case
      skip                 → discard the case
    """
    print(f"  [reflection {iteration}/{MAX_REFLECTIONS}] Error: {error_msg[:80]}")

    # ── Programmatic repair first (no API call needed) ────────────────────────
    if case.get("perturbation_type") == "flexion_angle":
        repaired = tool_repair_flexion_case(case, error_msg, output_dir)
        if repaired:
            return repaired

    # ── LLM reflection ────────────────────────────────────────────────────────
    try:
        keys = _load_keys()
    except Exception:
        print("  [reflection] Cannot load keys — skipping case")
        return None

    prompt = f"""
You are debugging an OpenSim spine musculoskeletal sensitivity analysis pipeline.
This is NOT a finite element simulation — only musculoskeletal / biomechanical.

A simulation case failed. Choose the best repair action.

FAILED CASE:
  label            : {case.get("label")}
  perturbation_type: {case.get("perturbation_type")}
  perturbation_value: {case.get("perturbation_value")}
  activity         : {case.get("activity_label")}
  error            : {error_msg}

REPAIR ACTIONS (choose ONE):
  "clamp_angle"          – Clamp flexion angle to [0, 90] degrees. Only valid
                           when perturbation_type = "flexion_angle".
  "use_default_activity" – Replace motion + force files with neutral standing
                           (activity 2). Use when custom .mot files are missing.
  "retry"                – Retry the same case unchanged. Use only for transient
                           subprocess errors (not file-not-found).
  "skip"                 – Discard this case. Use when the model file is invalid
                           or the error is unrecoverable.

OUTPUT: valid JSON only.
{{
  "action": "clamp_angle" | "use_default_activity" | "retry" | "skip",
  "reason": "<one sentence explanation>"
}}
"""

    try:
        content = ""
        if MODEL == "openai":
            client   = OpenAI(api_key=keys["openai_api_key"])
            response = client.chat.completions.create(
                model=MODEL_TYPE,
                messages=[
                    {"role": "system", "content": "Output strictly valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
        elif MODEL == "gemini":
            genai.configure(api_key=keys["gemini_api_key"])
            gem   = genai.GenerativeModel(MODEL_TYPE)
            resp  = gem.generate_content(prompt)
            content = resp.text.replace("```json", "").replace("```", "").strip()
        else:
            return None

        fix    = json.loads(content)
        action = fix.get("action", "skip")
        reason = fix.get("reason", "")
        print(f"  [reflection] LLM action → {action} | {reason}")

        if action == "skip":
            return None

        if action == "retry":
            return dict(case)

        if action == "clamp_angle":
            repaired = tool_repair_flexion_case(case, error_msg, output_dir)
            return repaired  # may be None if clamping doesn't help

        if action == "use_default_activity":
            return tool_use_default_activity(case)

        return None

    except Exception as exc:
        print(f"  [reflection] LLM call failed ({exc}) — skipping case")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESULTS AGGREGATION
# ════════════════════════════════════════════════════════════════════════════════

def _aggregate_sensitivity_results(
    results:          list,
    selected_models:  list,
) -> dict:
    """
    Convert raw sensitivity results into three extended DataFrames.

    Each DataFrame includes the columns:
      Perturbation_Type   — e.g. "flexion_angle", "slack_length_pct"
      Perturbation_Value  — numeric value (float)

    Returns dict with keys: "spinal", "forces", "activations"
    """
    model_map = {
        m["Filename"]: {
            "age":    (m.get("Min Age (year)", 0) + m.get("Max Age (year)", 0)) / 2,
            "weight": m.get("weight (kg)", 0),
            "height": m.get("height (m)", 0),
        }
        for m in selected_models
    }

    spinal_rows     = []
    force_rows      = []
    activation_rows = []

    for res in results:
        if res.get("status") != "Success":
            continue

        model_name = res.get("model", "")
        # Clean model name (strip temp suffixes) to match the original filename in model_map
        clean_model_name = model_name
        for possible_key in model_map:
            base_key = os.path.splitext(possible_key)[0]
            if base_key in model_name:
                clean_model_name = possible_key
                break

        demos = model_map.get(clean_model_name, {})

        base = {
            "Model":              model_name,
            "Activity":           res.get("activity_label", ""),
            "Perturbation_Type":  res.get("perturbation_type", ""),
            "Perturbation_Value": res.get("perturbation_value", None),
            "Age":                demos.get("age"),
            "Weight_kg":          demos.get("weight"),
            "Height_m":           demos.get("height"),
        }

        # Spinal loads
        sl = res.get("spinal_loads")
        if isinstance(sl, dict):
            for key, val in sl.items():
                if "IVDjnt" in key and any(s in key for s in ("_fx", "_fy", "_fz")):
                    row = base.copy()
                    row["Load_Name"] = key
                    row["Value"]     = val
                    spinal_rows.append(row)

        # Muscle forces
        mf = res.get("muscle_forces")
        if isinstance(mf, dict):
            for key, val in mf.items():
                row = base.copy()
                row["Muscle_Name"] = key
                row["Value"]       = val
                force_rows.append(row)

        # Muscle activations
        ma = res.get("muscle_activations")
        if isinstance(ma, dict):
            for key, val in ma.items():
                row = base.copy()
                row["Muscle_Name"] = key
                row["Value"]       = val
                activation_rows.append(row)

    # ── Spinal: pivot _fx / _fy / _fz into wide format ───────────────────────
    spinal_df_raw = pd.DataFrame(spinal_rows)
    if not spinal_df_raw.empty and "Load_Name" in spinal_df_raw.columns:
        ivd = spinal_df_raw[
            spinal_df_raw["Load_Name"].str.contains("IVDjnt", na=False)
        ].copy()
        if not ivd.empty:
            ivd["force_dir"] = ivd["Load_Name"].str.extract(r"_(fx|fy|fz)$", expand=False)
            ivd["load_base"] = ivd["Load_Name"].str.replace(r"_(fx|fy|fz)$", "", regex=True)
            ivd["Joint"]     = ivd["load_base"].str.split("_").str[:2].str.join("_")

            id_cols = [
                "Model", "Activity", "Perturbation_Type", "Perturbation_Value",
                "Age", "Weight_kg", "Height_m", "Joint", "load_base",
            ]
            pivot_cols = [c for c in id_cols if c in ivd.columns]

            wide = ivd.pivot_table(
                index=pivot_cols,
                columns="force_dir",
                values="Value",
                aggfunc="first",
            ).reset_index()

            wide.rename(
                columns={c: f"force_{c}" for c in ("fx", "fy", "fz") if c in wide.columns},
                inplace=True,
            )
            wide.columns.name = None
            wide.drop(columns=["load_base"], inplace=True, errors="ignore")
            spinal_final = wide
        else:
            spinal_final = ivd
    else:
        spinal_final = spinal_df_raw

    return {
        "spinal":      spinal_final,
        "forces":      pd.DataFrame(force_rows),
        "activations": pd.DataFrame(activation_rows),
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9 — LANGGRAPH NODES
# ════════════════════════════════════════════════════════════════════════════════

# ── Node 1: Planner ───────────────────────────────────────────────────────────

def sensitivity_planner_node(state: dict) -> dict:
    """
    Parse the user prompt into a structured sensitivity plan AND select the
    appropriate OpenSim model using the existing KNN/LLM model-selection logic.

    Writes to state:
      sensitivity_plan  — structured plan dict
      selected_models   — list of model info dicts (same format as normal pipeline)
      analysis_result   — subject filter (for downstream compatibility)
    """
    print("--- [Sensitivity] Node 1: Planner ---")
    user_prompt = state.get("user_prompt", "")

    # ── Model selection (reuse existing logic from utils.py) ──────────────────
    from utils import analyze_request_node, model_selection_node

    analysis_state = analyze_request_node({"user_prompt": user_prompt})
    if analysis_state.get("final_message"):
        return analysis_state

    model_result    = model_selection_node({
        "user_prompt":    user_prompt,
        "analysis_result": analysis_state.get("analysis_result", {}),
    })
    selected_models = model_result.get("selected_models", [])

    if not selected_models:
        return {"final_message": "Sensitivity Error: Could not select an OpenSim model."}

    # ── LLM plan extraction ───────────────────────────────────────────────────
    try:
        keys = _load_keys()
    except Exception as exc:
        return {"final_message": f"Config Error: {exc}"}

    tools_info = (
        f"1. 'flexion_sweep'         — trunk-flexion angle ± delta  "
        f"(valid range {FLEXION_MIN}°–{FLEXION_MAX}°, uses interpolation)\n"
        f"2. 'slack_length'          — scales all tendon_slack_length values ± pct%\n"
        f"3. 'vertebral_translation' — shifts all IVDjnt X-location ± mm\n"
        f"4. 'gravity'               — scales gravity Y ± pct%\n\n"
        f"Defaults: flexion ±{DEFAULT_FLEXION_DELTA}°, slack ±{DEFAULT_SLACK_PCT}%, "
        f"vert ±{DEFAULT_VERTEBRAL_MM} mm, gravity ±{DEFAULT_GRAVITY_PCT}%"
    )

    prompt = f"""
You are a biomechanics sensitivity analysis expert for OpenSim spine models
(musculoskeletal, NOT finite element).

USER PROMPT: "{user_prompt}"

AVAILABLE SENSITIVITY TOOLS:
{tools_info}

RULES:
• "ff-mode" or "sensitivity analysis" alone (no detail) → all 4 tools, all defaults.
• Only include a tool in the "tools" list if the user explicitly asks to vary/perturb that specific parameter (e.g. vary slack length, translate vertebral joints, change gravity, sweep flexion).
• A specific flexion task (e.g., "for 31 degree flexion", "during 20 deg flexion") defines the baseline activity (populate "flexion_activities" with that base_angle, delta_deg=0.0). Do NOT add "flexion_sweep" to the "tools" list unless the user explicitly asks to sweep/vary the flexion angle (e.g., "plus minus 5 degrees flexion").
• If the user mentions any custom/other parameter to vary (e.g. "pcsa", "optimal fiber length", etc.), extract its name as a tool and include it in the "tools" list (e.g., "pcsa"). Do NOT ignore it, do NOT fall back to defaults. Populate its value in the "custom_tool_values" map (key = tool name, value = float value e.g. 5.0 for 5 percent).
• include_base_model should always be true.

OUTPUT — valid JSON only:
{{
  "tools": ["flexion_sweep", "slack_length", "vertebral_translation", "gravity", "custom_parameter"],
  "flexion_activities": [
    {{"base_angle": 20, "delta_deg": 5, "weight_per_hand_kg": 0.0}}
  ],
  "slack_length_pct": 5.0,
  "vertebral_delta_mm": 2.0,
  "gravity_pct": 10.0,
  "include_base_model": true,
  "combine_tools": false,
  "custom_tool_values": {{
    "custom_parameter": 5.0
  }}
}}
"""

    try:
        content = ""
        if MODEL == "openai":
            client   = OpenAI(api_key=keys["openai_api_key"])
            response = client.chat.completions.create(
                model=MODEL_TYPE,
                messages=[
                    {"role": "system", "content": "Output strictly valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
        elif MODEL == "gemini":
            genai.configure(api_key=keys["gemini_api_key"])
            gem     = genai.GenerativeModel(MODEL_TYPE)
            resp    = gem.generate_content(prompt)
            content = resp.text.replace("```json", "").replace("```", "").strip()
        else:
            return {"final_message": f"Unknown MODEL: {MODEL}"}

        plan = json.loads(content)
        plan["user_prompt"] = user_prompt
        print(
            f"  Plan: tools={plan.get('tools')}\n"
            f"        flexion={plan.get('flexion_activities')}\n"
            f"        slack={plan.get('slack_length_pct')}% | "
            f"vert={plan.get('vertebral_delta_mm')} mm | "
            f"grav={plan.get('gravity_pct')}% | "
            f"custom={plan.get('custom_tool_values')}"
        )

        return {
            "sensitivity_plan":  plan,
            "selected_models":   selected_models,
            "analysis_result":   analysis_state.get("analysis_result", {}),
            "current_status":    f"Sensitivity plan ready. Tools: {plan.get('tools')}",
        }

    except Exception as exc:
        print(f"  Planner LLM Error: {exc}")
        return {"final_message": f"Sensitivity Planner Error: {exc}"}


# ── Node 2: Executor ──────────────────────────────────────────────────────────

def sensitivity_executor_node(state: dict) -> dict:
    """
    Agentic execution node for sensitivity analysis.

    Workflow
    --------
    1. Build ALL simulation cases (modified .osim files + .mot files)
       using the 12 executor tools below.
    2. Run each case sequentially via run_opensim_simulation().
    3. On failure: attempt programmatic repair (angle clamping) first, then
       ask the LLM for a repair action.  Up to MAX_REFLECTIONS (10) LLM calls.
    4. Cases that exceed the reflection budget are marked 'Skipped'.

    Executor Tools Available
    ------------------------
    [FILE TOOLS]
     1. generate_flexion_variants()         — create motion/force .mot files
     2. apply_slack_length_perturbation()   — XML-scale tendon_slack_length
     3. apply_vertebral_translation()       — XML-shift IVDjnt X-location
     4. apply_gravity_scale()               — XML-scale <gravity> Y
     5. apply_combined_perturbation()       — apply multiple XML mods at once

    [VALIDATION TOOLS]
     6. tool_validate_case()                — check required fields + file existence
     7. tool_validate_osim_xml()            — check modified .osim is valid XML
     8. tool_get_model_stats()              — count muscles/joints (verify mod)

    [FLOW-CONTROL TOOLS]
     9. tool_clamp_flexion_angle()          — clamp angle to [0°, 90°]
    10. tool_skip_case()                    — mark as Skipped + log
    11. tool_repair_flexion_case()          — auto-clamp + rebuild flexion .mot
    12. tool_use_default_activity()         — swap to neutral-standing motion/force
    13. tool_get_base_motion_force_paths()  — get neutral-standing relative paths
    14. tool_list_case_statuses()           — status summary dict
    15. _llm_reflect_on_error()             — LLM-guided repair (uses 1 reflection slot)
    """
    print("--- [Sensitivity] Node 2: Executor ---")

    plan            = state.get("sensitivity_plan", {})
    selected_models = state.get("selected_models", [])

    if not plan:
        return {"final_message": "Sensitivity Error: No plan in state."}
    if not selected_models:
        return {"final_message": "Sensitivity Error: No models selected."}

    reflection_count = 0
    error_log        = []
    all_results      = []

    # Keep track of the temp dir for repair tools
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.abspath(
        os.path.join(os.getcwd(), "opensim_files", TEMP_DIR_NAME, run_id)
    )

    # ── Step 1: Build all cases ───────────────────────────────────────────────
    print("  Building all sensitivity cases...")
    try:
        cases = _build_all_sensitivity_cases(plan, selected_models)
    except Exception as exc:
        return {"final_message": f"Sensitivity Case Build Error: {exc}"}

    if not cases:
        return {
            "final_message": (
                "Sensitivity: No simulation cases were generated.\n"
                "Check that at least one tool is in the plan and that a valid "
                "OpenSim model was selected."
            )
        }

    status = tool_list_case_statuses(cases, [])
    print(f"  {status['total']} cases to run.")

    # ── Step 2: Run all cases with LLM reflection on failure ──────────────────
    for case in cases:
        result = _run_single_sensitivity_case(case)

        if result["status"] == "Failed":
            if reflection_count < MAX_REFLECTIONS:
                reflection_count += 1
                fixed = _llm_reflect_on_error(
                    case=case,
                    error_msg=result.get("error", "unknown"),
                    plan=plan,
                    iteration=reflection_count,
                    output_dir=temp_dir,
                )

                if fixed:
                    print(f"  → Retry: {fixed.get('label', 'repaired')}")
                    result = _run_single_sensitivity_case(fixed)
                    error_log.append({
                        "original":     case["label"],
                        "error":        result.get("error", ""),
                        "fix":          fixed.get("label", ""),
                        "reflection":   reflection_count,
                        "retry_status": result["status"],
                    })
                else:
                    result = tool_skip_case(
                        case,
                        f"No fix found (reflection {reflection_count}): "
                        + result.get("error", ""),
                    )
                    error_log.append({
                        "original":   case["label"],
                        "error":      result.get("error", ""),
                        "fix":        "skipped",
                        "reflection": reflection_count,
                    })
            else:
                # Reflection budget exhausted
                result = tool_skip_case(
                    case,
                    f"Max reflections ({MAX_REFLECTIONS}) reached.",
                )

        all_results.append(result)

    # ── Step 3: Final summary ─────────────────────────────────────────────────
    status = tool_list_case_statuses(cases, all_results)
    summary = (
        f"Sensitivity execution complete.\n"
        f"  Success:     {status['success']}/{status['total']}\n"
        f"  Skipped:     {status['skipped']}\n"
        f"  Failed:      {status['failed']}\n"
        f"  Reflections: {reflection_count}/{MAX_REFLECTIONS}\n"
    )
    print(summary)

    return {
        "sensitivity_results":  all_results,
        "sensitivity_errors":   error_log,
        "sensitivity_iteration": reflection_count,
        "current_status":       summary,
    }


# ── Node 3: Analyst ───────────────────────────────────────────────────────────

def sensitivity_analyst_node(state: dict) -> dict:
    """
    Aggregate sensitivity results into extended DataFrames and produce an
    LLM-generated summary.

    Output DataFrames (stored in state["dataframes"]):
      "spinal"      — IVD joint loads, wide format (force_fx / force_fy / force_fz)
                      + Perturbation_Type + Perturbation_Value
      "forces"      — muscle forces (N)     + Perturbation_Type + Perturbation_Value
      "activations" — muscle activations    + Perturbation_Type + Perturbation_Value

    These DataFrames are fully compatible with the existing analysis_agent_node
    and can be used for further follow-up queries after the sensitivity run.
    """
    print("--- [Sensitivity] Node 3: Analyst ---")

    sens_results    = state.get("sensitivity_results", [])
    selected_models = state.get("selected_models", [])
    plan            = state.get("sensitivity_plan", {})
    error_log       = state.get("sensitivity_errors", [])

    if not sens_results:
        return {"final_message": "Sensitivity Analysis: No results to analyse."}

    # ── Build DataFrames ──────────────────────────────────────────────────────
    dfs = _aggregate_sensitivity_results(sens_results, selected_models)

    success_count = sum(1 for r in sens_results if r.get("status") == "Success")
    total_count   = len(sens_results)

    # ── Compute summary statistics per perturbation type ──────────────────────
    summary_stats: dict = {}
    spinal_df = dfs.get("spinal")
    if spinal_df is not None and not spinal_df.empty and "force_fy" in spinal_df.columns:
        try:
            spinal_df["force_fy"] = pd.to_numeric(spinal_df["force_fy"], errors="coerce")
            for ptype in spinal_df["Perturbation_Type"].unique():
                grp    = spinal_df[spinal_df["Perturbation_Type"] == ptype]
                l5s1   = grp[grp.get("Joint", pd.Series(dtype=str)).eq("L5_S1")]
                target = l5s1 if not l5s1.empty else grp
                summary_stats[ptype] = {
                    "L5_S1_comp_min_N":  round(float(target["force_fy"].min()), 1),
                    "L5_S1_comp_max_N":  round(float(target["force_fy"].max()), 1),
                    "n_variants":        int(target["Perturbation_Value"].nunique()),
                }
        except Exception as exc:
            print(f"  [analyst] stats error: {exc}")

    # ── LLM summary ──────────────────────────────────────────────────────────
    tools_used = plan.get("tools", [])
    llm_summary = ""
    try:
        keys = _load_keys()
        prompt = f"""
You are a biomechanics expert. Write a concise sensitivity analysis summary
(3-5 sentences, clinical tone) for an OpenSim spine musculoskeletal model.

Simulation: {success_count}/{total_count} cases succeeded.
Tools used: {tools_used}
L5-S1 compression results by perturbation type:
{json.dumps(summary_stats, indent=2)}
Errors/reflections logged: {len(error_log)}

Include:
1. Which parameters were varied
2. Range of L5-S1 compression forces observed
3. Which perturbation had the largest effect (if identifiable)
4. Any caveats (failed runs, clamping applied)
"""
        content = ""
        if MODEL == "openai":
            client   = OpenAI(api_key=keys["openai_api_key"])
            response = client.chat.completions.create(
                model=MODEL_TYPE,
                messages=[
                    {"role": "system", "content": "You are a biomechanics expert."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content
        elif MODEL == "gemini":
            genai.configure(api_key=keys["gemini_api_key"])
            gem     = genai.GenerativeModel(MODEL_TYPE)
            resp    = gem.generate_content(prompt)
            content = resp.text.strip()

        llm_summary = content
    except Exception as exc:
        llm_summary = f"(LLM summary unavailable: {exc})"

    # ── Final message ─────────────────────────────────────────────────────────
    rows_spinal = len(dfs.get("spinal", pd.DataFrame()))
    rows_forces = len(dfs.get("forces", pd.DataFrame()))
    rows_activ  = len(dfs.get("activations", pd.DataFrame()))

    final_message = (
        f"## Sensitivity Analysis Complete\n\n"
        f"**Runs:** {success_count}/{total_count} succeeded "
        f"| **Tools:** {', '.join(tools_used)}\n\n"
        f"### Clinical Summary\n{llm_summary}\n\n"
        f"### DataFrames Available for Follow-up Queries\n"
        f"| DataFrame | Rows | Key columns |\n"
        f"|-----------|------|-------------|\n"
        f"| `spinal`      | {rows_spinal} | Joint, force_fx/fy/fz, Perturbation_Type, Perturbation_Value |\n"
        f"| `forces`      | {rows_forces} | Muscle_Name, Value (N), Perturbation_Type, Perturbation_Value |\n"
        f"| `activations` | {rows_activ}  | Muscle_Name, Value (0–1), Perturbation_Type, Perturbation_Value |\n"
    )
    if error_log:
        final_message += f"\n> **{len(error_log)} reflection(s)** were used during execution.\n"

    return {
        "dataframes":    dfs,
        "final_message": final_message,
        "current_status": f"Sensitivity analysis done. {success_count}/{total_count} runs OK.",
    }
