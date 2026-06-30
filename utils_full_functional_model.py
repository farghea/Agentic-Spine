"""
utils_full_functional_model.py
-------------------------------
XML-based OpenSim model modification utilities for the Sensitivity Analysis
(FF-Mode) pipeline.

All functions produce *modified copies* of .osim files without touching the
originals.  No OpenSim Python API is required — modifications are done via
regex/string operations on the raw file text.

Why text-based and NOT xml.etree.ElementTree parse-and-write?
  Parsing + re-serialising a 2 MB OpenSim file with ElementTree reformats every
  tag, changes whitespace, and can silently alter numeric precision — which can
  break the OpenSim 3.0 parser or the downstream analyses.  Text-level regex is
  surgical and leaves 100 % of the rest of the file untouched.

Verified against real NMB spine models (~2 MB, OpenSimDocument Version=30000):
  • <gravity>          — single tag, line 9, format "0 -9.8066 0"
  • <tendon_slack_length> — 552 occurrences, always single-line float
  • <CustomJoint name="*_IVDjnt"> / <location_in_parent> — 18 IVD joints,
    format "X Y Z" (may span two lines)
"""

import os
import re
import shutil


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def copy_osim_to_temp(osim_path: str, output_dir: str, suffix: str) -> str:
    """
    Copy *osim_path* to *output_dir* with a suffix inserted before the
    extension, e.g.  "110_...MuscleAdjust_slack_plus5pct.osim".

    Returns the absolute path to the new copy.
    """
    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(osim_path))[0]
    out_name = f"{base}_{suffix}.osim"
    out_path = os.path.join(output_dir, out_name)
    shutil.copy2(osim_path, out_path)
    return out_path


def validate_osim_xml(osim_path: str) -> tuple:
    """
    Quick validation that *osim_path* is still parseable XML.

    Returns (True, "") on success or (False, error_message) on failure.
    Note: validation uses ElementTree *only* for reading, never for writing,
    so no reformatting occurs.
    """
    try:
        import xml.etree.ElementTree as ET
        ET.parse(osim_path)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def get_osim_stats(osim_path: str) -> dict:
    """
    Return basic statistics about the model for verification / debugging.

    Keys returned:
      tendon_slack_count  — number of <tendon_slack_length> tags found
      ivdjnt_count        — number of *_IVDjnt CustomJoint declarations
      gravity             — raw text content of <gravity> tag
      file_size_kb        — file size in kB
    """
    try:
        with open(osim_path, "r", encoding="utf-8") as f:
            content = f.read()

        slack_count  = len(re.findall(r"<tendon_slack_length>", content))
        ivdjnt_count = len(re.findall(r'CustomJoint name="[^"]*_IVDjnt"', content))
        grav_match   = re.search(r"<gravity>([\d.\s\-eE+]+)</gravity>", content)
        gravity_val  = grav_match.group(1).strip() if grav_match else "not found"

        return {
            "tendon_slack_count": slack_count,
            "ivdjnt_count":       ivdjnt_count,
            "gravity":            gravity_val,
            "file_size_kb":       os.path.getsize(osim_path) // 1024,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ════════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Muscle Passive Properties / Slack Length
# ════════════════════════════════════════════════════════════════════════════════

def apply_slack_length_perturbation(
    osim_path:   str,
    scale:       float,
    output_path: str,
) -> str:
    """
    Scale *all* ``<tendon_slack_length>`` values by *scale*.

    Parameters
    ----------
    osim_path   : source .osim file path
    scale       : multiplicative factor (e.g. 1.05 = +5 %, 0.95 = −5 %)
    output_path : where to write the modified copy

    Returns
    -------
    output_path  (for chaining)

    Notes
    -----
    The regex targets only the inner float text of the tag, leaving surrounding
    whitespace, comments, and all other tags completely untouched.  The NMB
    spine models have exactly 552 such tags.
    """
    with open(osim_path, "r", encoding="utf-8") as f:
        content = f.read()

    count = [0]

    def _scale(m):
        count[0] += 1
        val = float(m.group(1))
        return f"<tendon_slack_length>{val * scale:.10f}</tendon_slack_length>"

    modified = re.sub(
        r"<tendon_slack_length>([\d.eE+\-]+)</tendon_slack_length>",
        _scale,
        content,
    )

    _write(output_path, modified)
    print(
        f"  [slack_length] scale={scale:.4f} — "
        f"modified {count[0]} entries → {os.path.basename(output_path)}"
    )
    return output_path


# ════════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Vertebral Joint Translation (X-axis = anterior–posterior)
# ════════════════════════════════════════════════════════════════════════════════

def apply_vertebral_translation(
    osim_path:    str,
    delta_m_x:    float,
    output_path:  str,
) -> str:
    """
    Shift the **X component** of ``<location_in_parent>`` for every
    ``*_IVDjnt`` CustomJoint in the model.

    X is the anterior–posterior direction in the NMB spine model.
    All 18 intervertebral disc joints (L5_S1 → T1_T2) are displaced
    simultaneously by the *same* delta.

    Parameters
    ----------
    osim_path    : source .osim file path
    delta_m_x    : translation in **metres** (+ve = anterior, −ve = posterior)
                   e.g. +0.002 for +2 mm
    output_path  : where to write the modified copy

    Returns
    -------
    output_path  (for chaining)

    Implementation
    --------------
    Finds each ``<CustomJoint name="..._IVDjnt">`` tag, then locates the
    *first* ``<location_in_parent>`` that follows it (the joint's own placement
    tag) and shifts the X coordinate.  Because we use a non-greedy match up to
    ``</CustomJoint>``, inner nested tags are not misidentified.

    The values may span two lines in some models, so the regex uses
    ``[\\s\\S]*?`` for the XYZ capture and ``re.DOTALL``.
    """
    with open(osim_path, "r", encoding="utf-8") as f:
        content = f.read()

    modified_count = [0]

    # Pattern explanation:
    #   group 1 — CustomJoint opening tag for an IVDjnt joint
    #   group 2 — everything between the opening tag and location_in_parent
    #   group 3 — <location_in_parent> open tag
    #   group 4 — XYZ values (may be multi-line, only digits / sign / white)
    #   group 5 — </location_in_parent> close tag
    PATTERN = re.compile(
        r"(CustomJoint name=\"[^\"]*_IVDjnt\"[^>]*>)"
        r"([\s\S]*?)"
        r"(<location_in_parent>)"
        r"([\d.\s\-+eE\n\r]+?)"
        r"(</location_in_parent>)",
        re.DOTALL,
    )

    def _shift(m):
        xyz_raw = m.group(4).strip()
        parts   = xyz_raw.split()
        if len(parts) < 3:
            return m.group(0)  # unexpected format — leave untouched

        x = float(parts[0]) + delta_m_x
        y = float(parts[1])
        z = float(parts[2])
        modified_count[0] += 1
        return (
            m.group(1)
            + m.group(2)
            + m.group(3)
            + f" {x:.10f} {y:.10f} {z:.10f} "
            + m.group(5)
        )

    modified = PATTERN.sub(_shift, content)

    _write(output_path, modified)
    print(
        f"  [vertebral_translation] delta_x={delta_m_x*1000:.2f} mm — "
        f"modified {modified_count[0]} IVDjnt joints → {os.path.basename(output_path)}"
    )
    return output_path


# ════════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Gravity Scaling
# ════════════════════════════════════════════════════════════════════════════════

def apply_gravity_scale(
    osim_path:     str,
    gravity_scale: float,
    output_path:   str,
) -> str:
    """
    Scale the **Y component** of ``<gravity>`` by *gravity_scale*.

    Default OpenSim gravity: ``0 -9.8066 0`` (Y is vertical, downward).
    X and Z are always 0 in this model and are preserved unchanged.

    Parameters
    ----------
    osim_path      : source .osim file path
    gravity_scale  : multiplier for the Y gravity value
                     (e.g. 1.10 = +10 %, 0.90 = −10 %)
    output_path    : where to write the modified copy

    Returns
    -------
    output_path  (for chaining)
    """
    with open(osim_path, "r", encoding="utf-8") as f:
        content = f.read()

    count = [0]

    def _scale_gravity(m):
        parts = m.group(1).split()
        if len(parts) < 3:
            return m.group(0)
        gx = float(parts[0])
        gy = float(parts[1]) * gravity_scale
        gz = float(parts[2])
        count[0] += 1
        return f"<gravity> {gx:.6f} {gy:.6f} {gz:.6f}</gravity>"

    modified = re.sub(
        r"<gravity>([\d.\s\-+eE]+)</gravity>",
        _scale_gravity,
        content,
    )

    if count[0] == 0:
        print(f"  [gravity_scale] WARNING: <gravity> tag not found in {osim_path}")

    _write(output_path, modified)
    print(
        f"  [gravity_scale] scale={gravity_scale:.4f} — "
        f"gravity tag modified → {os.path.basename(output_path)}"
    )
    return output_path


# ════════════════════════════════════════════════════════════════════════════════
# COMBINED — Apply multiple perturbations in one pass
# ════════════════════════════════════════════════════════════════════════════════

def apply_combined_perturbation(
    osim_path:          str,
    output_path:        str,
    slack_scale:        float = 1.0,
    vertebral_delta_m:  float = 0.0,
    gravity_scale:      float = 1.0,
) -> str:
    """
    Apply multiple perturbations to a single model sequentially.

    Useful for combinatorial sensitivity (e.g. slack + gravity at once).
    Temporary intermediate files are created in the same directory as
    *output_path* and cleaned up when done.

    Returns output_path.
    """
    temp_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(temp_dir, exist_ok=True)

    current = osim_path
    tmp_files = []

    if abs(slack_scale - 1.0) > 1e-9:
        tmp = os.path.join(temp_dir, "_comb_tmp_slack.osim")
        apply_slack_length_perturbation(current, slack_scale, tmp)
        current = tmp
        tmp_files.append(tmp)

    if abs(vertebral_delta_m) > 1e-9:
        tmp = os.path.join(temp_dir, "_comb_tmp_vert.osim")
        apply_vertebral_translation(current, vertebral_delta_m, tmp)
        current = tmp
        tmp_files.append(tmp)

    if abs(gravity_scale - 1.0) > 1e-9:
        tmp = os.path.join(temp_dir, "_comb_tmp_grav.osim")
        apply_gravity_scale(current, gravity_scale, tmp)
        current = tmp
        tmp_files.append(tmp)

    # Copy final result to the requested output path
    if current == osim_path:
        shutil.copy2(osim_path, output_path)
    else:
        shutil.copy2(current, output_path)

    # Clean up intermediates (keep output_path)
    for fp in tmp_files:
        if os.path.exists(fp) and os.path.abspath(fp) != os.path.abspath(output_path):
            try:
                os.remove(fp)
            except OSError:
                pass

    return output_path


# ════════════════════════════════════════════════════════════════════════════════
# PRIVATE WRITE HELPER
# ════════════════════════════════════════════════════════════════════════════════

def _write(path: str, content: str) -> None:
    """Create parent dirs as needed and write *content* as UTF-8."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
