"""
interpolate_mot.py
------------------
Interpolates OpenSim .mot files based on flexion angle.

Rules:
- Lines 1-7  (header):   kept exactly from the reference file, only the
                          motion name on line 1 is replaced.
- Lines 8-18 (data rows): column values are linearly interpolated across the
                           five reference flexion angles using numpy.

Reference flexion map (source files are in the 'flexion/' subfolder):
    0°  -> NMB_Motion2.mot
    10° -> NMB_Motion28.mot
    30° -> NMB_Motion3.mot
    45° -> NMB_Motion4.mot
    90° -> NMB_Motion5.mot

Usage
-----
Call `generate_interpolated_mot(flexion_angles, output_dir)` with a list of
desired flexion angles (degrees) and an output directory path.

Example:
    from interpolate_mot import generate_interpolated_mot

    generate_interpolated_mot(
        flexion_angles=[0, 5, 10, 20, 30, 45, 60, 90],
        output_dir=r"C:/path/to/output"
    )
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder containing the five reference .mot files
FLEXION_DIR = os.path.join(os.path.dirname(__file__), "flexion")

# Reference flexion angle -> filename mapping
REFERENCE_FILES = {
    0:  "NMB_Motion2.mot",
    10: "NMB_Motion28.mot",
    30: "NMB_Motion3.mot",
    45: "NMB_Motion4.mot",
    90: "NMB_Motion5.mot",
}

# Sorted reference angles (used for interpolation)
REF_ANGLES = sorted(REFERENCE_FILES.keys())   # [0, 10, 30, 45, 90]


# ---------------------------------------------------------------------------
# Helper: parse a single .mot file
# ---------------------------------------------------------------------------

def _parse_mot(filepath):
    """
    Returns:
        header_lines : list[str]  – lines 1-7 (0-indexed 0-6), raw text
        data_rows    : np.ndarray – shape (n_rows, n_cols), float64
    """
    with open(filepath, "r") as f:
        raw_lines = f.readlines()

    # Lines 1-7  → indices 0-6  (header)
    header_lines = raw_lines[:7]

    # Lines 8-18 → indices 7-17 (data), skip blank trailing lines
    data_lines = [l for l in raw_lines[7:] if l.strip()]
    data_rows = np.array(
        [list(map(float, l.split())) for l in data_lines],
        dtype=np.float64
    )
    return header_lines, data_rows


# ---------------------------------------------------------------------------
# Core: load all reference files once
# ---------------------------------------------------------------------------

def _load_references():
    """Load and return {angle: (header_lines, data_rows)} for all references."""
    refs = {}
    for angle, fname in REFERENCE_FILES.items():
        fpath = os.path.join(FLEXION_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Reference file not found: {fpath}")
        header, data = _parse_mot(fpath)
        refs[angle] = (header, data)
    return refs


# ---------------------------------------------------------------------------
# Core: interpolate data rows at a given flexion angle
# ---------------------------------------------------------------------------

def _interpolate_data(target_angle, refs):
    """
    Linear interpolation of data rows at target_angle.

    If target_angle exactly matches a reference, that reference is returned.
    If target_angle is outside [0, 90] a ValueError is raised.
    """
    if not (REF_ANGLES[0] <= target_angle <= REF_ANGLES[-1]):
        raise ValueError(
            f"Flexion angle {target_angle} is outside the valid range "
            f"[{REF_ANGLES[0]}, {REF_ANGLES[-1]}]."
        )

    # Exact match
    if target_angle in refs:
        return refs[target_angle][1].copy()

    # Find surrounding reference angles
    lo = max(a for a in REF_ANGLES if a <= target_angle)
    hi = min(a for a in REF_ANGLES if a >= target_angle)

    t = (target_angle - lo) / (hi - lo)          # interpolation factor [0,1]
    data_lo = refs[lo][1]
    data_hi = refs[hi][1]

    return (1.0 - t) * data_lo + t * data_hi


# ---------------------------------------------------------------------------
# Core: write interpolated .mot file
# ---------------------------------------------------------------------------

def _write_mot(output_path, motion_name, ref_header, interp_data):
    """
    Write a valid OpenSim .mot file:
        Line 1  : motion name
        Lines 2-7: rest of header (version, nRows, nColumns, ...)
        Lines 8-18: interpolated data rows
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="\r\n") as f:
        # Line 1: motion name (replace original name)
        f.write(motion_name + "\n")

        # Lines 2-7: rest of the header unchanged
        for line in ref_header[1:]:
            f.write(line.rstrip("\r\n") + "\n")

        # Lines 8-18: data rows
        for row in interp_data:
            formatted = "\t".join(f"{v:18.8f}" for v in row)
            f.write(formatted + "\n")

        f.write("\n")   # trailing blank line (matches original format)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_interpolated_mot(flexion_angles, output_dir=None, name_prefix="NMB_Motion_flex"):
    """
    Generate interpolated .mot files for each angle in `flexion_angles`.

    Parameters
    ----------
    flexion_angles : list[float]
        Desired flexion angles in degrees. Must be within [0, 90].
    output_dir : str or None
        Directory where the output files are saved.
        Defaults to the same folder as this script
        (i.e. 'motion force interpolation/').
    name_prefix : str
        Prefix for the output file names.
        Files are named  <name_prefix><angle>deg.mot

    Returns
    -------
    list[str]
        Absolute paths of all files that were written.
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    refs = _load_references()
    # Use the angle-0 header as the template
    template_header = refs[0][0]

    written = []
    for angle in flexion_angles:
        interp_data = _interpolate_data(angle, refs)
        motion_name = f"{name_prefix}{int(angle)}deg"
        filename    = f"{motion_name}.mot"
        out_path    = os.path.join(output_dir, filename)

        _write_mot(out_path, motion_name, template_header, interp_data)
        written.append(out_path)
        print(f"  [OK] {filename}")

    print(f"\nDone — {len(written)} file(s) written to:\n  {os.path.abspath(output_dir)}")
    return written


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(output_dir=None):
    """
    Quick sanity-check:
    1. Reference angles produce data identical to the source files.
    2. An in-between angle (e.g. 20°) is truly between its brackets.
    3. All generated files are valid .mot files (parseable).
    """
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    refs = _load_references()
    passed = 0
    failed = 0

    # -- Test 1: exact reference angles reproduce source data --
    print("\n[Test 1] Exact reference angles reproduce source data")
    for angle in REF_ANGLES:
        interp = _interpolate_data(angle, refs)
        source = refs[angle][1]
        if np.allclose(interp, source, atol=1e-10):
            print(f"  PASS  flexion={angle:3d}°")
            passed += 1
        else:
            print(f"  FAIL  flexion={angle:3d}°  max_diff={np.max(np.abs(interp - source)):.2e}")
            failed += 1

    # -- Test 2: intermediate angle is truly between brackets --
    print("\n[Test 2] Intermediate values lie between reference brackets")
    test_cases = [(5, 0, 10), (20, 10, 30), (37.5, 30, 45), (67.5, 45, 90)]
    for angle, lo, hi in test_cases:
        interp = _interpolate_data(angle, refs)
        lo_data = refs[lo][1]
        hi_data = refs[hi][1]
        within = np.all(
            (interp >= np.minimum(lo_data, hi_data) - 1e-9) &
            (interp <= np.maximum(lo_data, hi_data) + 1e-9)
        )
        status = "PASS" if within else "FAIL"
        if within:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  flexion={angle}°  (between {lo}° and {hi}°)")

    # -- Test 3: generated files are parseable .mot files --
    if output_dir and os.path.isdir(output_dir):
        print("\n[Test 3] Generated files are valid .mot files")
        for fname in os.listdir(output_dir):
            if fname.endswith(".mot") and fname not in REFERENCE_FILES.values():
                fpath = os.path.join(output_dir, fname)
                try:
                    header, data = _parse_mot(fpath)
                    assert len(header) == 7,   "Expected 7 header lines"
                    assert data.shape[1] == 151, "Expected 151 columns"
                    print(f"  PASS  {fname}  shape={data.shape}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {fname}  error={e}")
                    failed += 1
    else:
        print("\n[Test 3] Skipped (no output_dir provided or directory missing)")

    # -- Summary --
    print(f"\nResult: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


# ---------------------------------------------------------------------------
# Entry point (run directly to demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

    # Example: generate files for these flexion angles
    angles_to_generate = [0, 5, 10, 20, 30, 45, 60, 75, 90]

    print("Generating interpolated .mot files ...")
    generate_interpolated_mot(
        flexion_angles=angles_to_generate,
        output_dir=OUTPUT_DIR,
    )

    # Run verification
    verify(output_dir=OUTPUT_DIR)
