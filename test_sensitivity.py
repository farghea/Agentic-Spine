#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sensitivity.py
-------------------
Test suite for the Sensitivity Analysis / FF-Mode pipeline.
ASCII-only output so it works under any console encoding (Windows cp1252).

Covers
------
  UNIT  -- fast, no API calls, no OpenSim subprocess
    U1  keyword detection (is_sensitivity_prompt)
    U2  model path resolution (opensim_files/ prefix logic)
    U3  XML: apply_slack_length_perturbation
    U4  XML: apply_vertebral_translation
    U5  XML: apply_gravity_scale
    U6  validate_osim_xml on modified files
    U7  get_osim_stats on modified files
    U8  tool_clamp_flexion_angle edge cases
    U9  tool_validate_case field checks

  INTEGRATION -- slow, requires conda a4s env + OpenSim conda env
    I1  Flexion sweep  (Tool 1 only)
    I2  Slack length   (Tool 2 only)
    I3  Vertebral translation (Tool 3 only)
    I4  Gravity        (Tool 4 only)
    I5  FF-mode bare   (all 4 tools, defaults)
    I6  Flexion + hand load  (extra case)

USAGE (from project root, inside the a4s environment)
------
    python test_sensitivity.py              # all tests
    python test_sensitivity.py --unit       # unit tests only (~5 s, no API)
    python test_sensitivity.py --integ      # integration tests only
    python test_sensitivity.py --id I1      # single integration test
    python test_sensitivity.py --id U3      # single unit test group
"""

import os
import sys
import re
import shutil
import time
import json
import argparse
import traceback

# Always run from project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Force stdout to UTF-8 so emoji/unicode don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Print helpers (ASCII-only labels)
# ---------------------------------------------------------------------------
def _ok(msg):    print(f"  [PASS] {msg}")
def _fail(msg, exc=None):
    print(f"  [FAIL] {msg}")
    if exc:
        for line in str(exc).splitlines()[:5]:
            print(f"         {line}")
def _info(msg):  print(f"  [INFO] {msg}")
def _warn(msg):  print(f"  [WARN] {msg}")
def _hdr(msg):
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {msg}")
    print(bar)
def _sub(msg):
    print(f"\n  -- {msg} --")


# ---------------------------------------------------------------------------
# Global result tracker
# ---------------------------------------------------------------------------
RESULTS = []   # list of (test_id, name, passed, notes)

def _record(tid, name, passed, notes=""):
    RESULTS.append((tid, name, bool(passed), notes))
    if passed:
        _ok(f"[{tid}] {name}")
    else:
        _fail(f"[{tid}] {name}", notes if notes else None)
    return bool(passed)

def _assert(condition, tid, name, notes=""):
    return _record(tid, name, bool(condition), notes if not condition else "")


# ---------------------------------------------------------------------------
# Helper: locate a real .osim file
# ---------------------------------------------------------------------------
def _find_osim() -> str:
    root = os.path.join(PROJECT_DIR, "opensim_files")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".osim"):
                return os.path.join(dirpath, f)
    raise FileNotFoundError("No .osim found under opensim_files/")


# ===========================================================================
# UNIT TESTS
# ===========================================================================

def run_unit_tests(filter_id=None):
    _hdr("UNIT TESTS  (no API, no OpenSim)")

    from utils_sensitivity import (
        is_sensitivity_prompt,
        tool_clamp_flexion_angle,
        tool_validate_case,
    )
    from utils_full_functional_model import (
        apply_slack_length_perturbation,
        apply_vertebral_translation,
        apply_gravity_scale,
        validate_osim_xml,
        get_osim_stats,
    )

    tmp_dir = os.path.join(PROJECT_DIR, "_test_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    def _skip(tid):
        return filter_id and not tid.startswith(filter_id.rstrip("0123456789"))

    # -------------------------------------------------------------------
    # U1: Keyword detection
    # -------------------------------------------------------------------
    if not _skip("U1"):
        _sub("U1 -- Keyword detection")

        triggers = [
            "Sensitivity analysis on 30 degree flexion",
            "Do a sensitivity study on this model",
            "ff-mode for a 70 kg male",
            "full flexibility mode",
            "parametric analysis on gravity",
            "perturbation study on slack length",
            "SENSITIVITY ANALYSIS upper case",
        ]
        non_triggers = [
            "Simulate standing upright for a 72 kg man",
            "What is the compression force at L5-S1?",
            "30 degree trunk flexion for a female",
            "Show me the muscle forces",
        ]

        bad_trig    = [t for t in triggers     if not is_sensitivity_prompt(t)]
        bad_nontrig = [t for t in non_triggers if     is_sensitivity_prompt(t)]

        _assert(not bad_trig,    "U1a", "All trigger phrases detected",
                str(bad_trig) if bad_trig else "")
        _assert(not bad_nontrig, "U1b", "Non-sensitivity prompts not triggered",
                str(bad_nontrig) if bad_nontrig else "")

    # -------------------------------------------------------------------
    # U2: Model path resolution
    # -------------------------------------------------------------------
    if not _skip("U2"):
        _sub("U2 -- Model path resolution (opensim_files/ prefix)")

        opensim_files_dir = os.path.join(PROJECT_DIR, "opensim_files")
        sample_rel = None
        for dirpath, _, files in os.walk(opensim_files_dir):
            for f in files:
                if f.endswith(".osim"):
                    rel = os.path.relpath(os.path.join(dirpath, f), opensim_files_dir)
                    sample_rel = rel.replace("\\", "/")
                    break
            if sample_rel:
                break

        if sample_rel:
            resolved = os.path.join(opensim_files_dir, sample_rel)
            _assert(os.path.exists(resolved), "U2a",
                    f"Relative path '{sample_rel}' resolves via opensim_files/ prefix")
        else:
            _warn("U2a skipped -- no .osim found")

    # -------------------------------------------------------------------
    # U3: Slack length perturbation
    # -------------------------------------------------------------------
    if not _skip("U3"):
        _sub("U3 -- apply_slack_length_perturbation")
        try:
            osim  = _find_osim()
            out_p = os.path.join(tmp_dir, "test_slack_plus5.osim")
            out_m = os.path.join(tmp_dir, "test_slack_minus5.osim")

            apply_slack_length_perturbation(osim, 1.05, out_p)
            apply_slack_length_perturbation(osim, 0.95, out_m)

            tag = r"<tendon_slack_length>([\d.eE+\-]+)</tendon_slack_length>"
            with open(osim,  "r") as f: orig_txt = f.read()
            with open(out_p, "r") as f: mod_p    = f.read()
            with open(out_m, "r") as f: mod_m    = f.read()

            orig_v = [float(x) for x in re.findall(tag, orig_txt)]
            mod_pv = [float(x) for x in re.findall(tag, mod_p)]
            mod_mv = [float(x) for x in re.findall(tag, mod_m)]

            _assert(len(orig_v) > 0 and len(orig_v) == len(mod_pv),
                    "U3a", f"tendon_slack_length count preserved ({len(orig_v)})")

            ratios_p = [m / o for o, m in zip(orig_v, mod_pv) if o != 0]
            ratios_m = [m / o for o, m in zip(orig_v, mod_mv) if o != 0]

            bad_p = sum(1 for r in ratios_p if abs(r - 1.05) > 1e-6)
            bad_m = sum(1 for r in ratios_m if abs(r - 0.95) > 1e-6)

            _assert(bad_p == 0, "U3b", f"All {len(ratios_p)} values scaled x1.05 (+5%)",
                    f"{bad_p} values not correctly scaled")
            _assert(bad_m == 0, "U3c", f"All {len(ratios_m)} values scaled x0.95 (-5%)",
                    f"{bad_m} values not correctly scaled")

        except Exception as exc:
            _record("U3", "apply_slack_length_perturbation", False, traceback.format_exc())

    # -------------------------------------------------------------------
    # U4: Vertebral translation
    # -------------------------------------------------------------------
    if not _skip("U4"):
        _sub("U4 -- apply_vertebral_translation")
        try:
            osim    = _find_osim()
            delta_m = 0.002    # +2 mm
            out     = os.path.join(tmp_dir, "test_vert_plus2mm.osim")
            apply_vertebral_translation(osim, delta_m, out)

            with open(osim, "r") as f: orig_txt = f.read()
            with open(out,  "r") as f: mod_txt  = f.read()

            orig_ivd = len(re.findall(r'CustomJoint name="[^"]*_IVDjnt"', orig_txt))
            mod_ivd  = len(re.findall(r'CustomJoint name="[^"]*_IVDjnt"', mod_txt))

            _assert(orig_ivd > 0, "U4a", f"Original has {orig_ivd} IVDjnt joints")
            _assert(orig_ivd == mod_ivd, "U4b", "IVDjnt count unchanged after modification")
            _assert(orig_txt != mod_txt,  "U4c", "Modified file differs from original")

            # Spot-check L5_S1 X shift
            pat = (r'CustomJoint name="L5_S1_IVDjnt"[\s\S]*?'
                   r'<location_in_parent>([\d.\s\-+eE\n\r]+?)</location_in_parent>')
            mo  = re.search(pat, orig_txt)
            mm  = re.search(pat, mod_txt)
            if mo and mm:
                xo = float(mo.group(1).strip().split()[0])
                xm = float(mm.group(1).strip().split()[0])
                shifted = abs((xm - xo) - delta_m) < 1e-7
                _assert(shifted, "U4d",
                        f"L5_S1 X shifted by {delta_m*1000:.1f} mm "
                        f"({xo:.6f} -> {xm:.6f})",
                        f"Expected shift {delta_m}, got {xm-xo:.8f}")
            else:
                _warn("U4d skipped -- could not extract L5_S1_IVDjnt X")

        except Exception as exc:
            _record("U4", "apply_vertebral_translation", False, traceback.format_exc())

    # -------------------------------------------------------------------
    # U5: Gravity scale
    # -------------------------------------------------------------------
    if not _skip("U5"):
        _sub("U5 -- apply_gravity_scale")
        try:
            osim  = _find_osim()
            out_p = os.path.join(tmp_dir, "test_grav_plus10.osim")
            out_m = os.path.join(tmp_dir, "test_grav_minus10.osim")

            apply_gravity_scale(osim, 1.10, out_p)
            apply_gravity_scale(osim, 0.90, out_m)

            def _grav_y(path):
                with open(path, "r") as f: c = f.read()
                m = re.search(r"<gravity>([\d.\s\-+eE]+)</gravity>", c)
                if m:
                    parts = m.group(1).split()
                    return float(parts[1]) if len(parts) >= 2 else None
                return None

            gy_o = _grav_y(osim)
            gy_p = _grav_y(out_p)
            gy_m = _grav_y(out_m)

            _assert(gy_o is not None, "U5a", f"Original gravity Y found: {gy_o}")
            _assert(gy_p is not None and abs(gy_p / gy_o - 1.10) < 1e-5,
                    "U5b", f"+10% gravity: {gy_o:.5f} -> {gy_p:.5f}",
                    f"Ratio = {gy_p/gy_o if gy_p else 'N/A'}")
            _assert(gy_m is not None and abs(gy_m / gy_o - 0.90) < 1e-5,
                    "U5c", f"-10% gravity: {gy_o:.5f} -> {gy_m:.5f}",
                    f"Ratio = {gy_m/gy_o if gy_m else 'N/A'}")

        except Exception as exc:
            _record("U5", "apply_gravity_scale", False, traceback.format_exc())

    # -------------------------------------------------------------------
    # U6: XML validity after modifications
    # -------------------------------------------------------------------
    if not _skip("U6"):
        _sub("U6 -- validate_osim_xml on modified files")
        for fname in [
            "test_slack_plus5.osim", "test_slack_minus5.osim",
            "test_vert_plus2mm.osim",
            "test_grav_plus10.osim", "test_grav_minus10.osim",
        ]:
            path = os.path.join(tmp_dir, fname)
            if os.path.exists(path):
                valid, err = validate_osim_xml(path)
                _assert(valid, f"U6/{fname.split('.')[0][-10:]}", f"Valid XML: {fname}", err)
            else:
                _warn(f"U6/{fname} -- file not found (prior test may have failed)")

    # -------------------------------------------------------------------
    # U7: get_osim_stats
    # -------------------------------------------------------------------
    if not _skip("U7"):
        _sub("U7 -- get_osim_stats")
        try:
            osim  = _find_osim()
            stats = get_osim_stats(osim)
            _assert(stats.get("tendon_slack_count", 0) > 0, "U7a",
                    f"tendon_slack_count = {stats.get('tendon_slack_count')}")
            _assert(stats.get("ivdjnt_count", 0) > 0, "U7b",
                    f"ivdjnt_count = {stats.get('ivdjnt_count')}")
            _assert("gravity" in stats, "U7c",
                    f"gravity field = '{stats.get('gravity')}'")
            _info(f"Full stats: {stats}")
        except Exception as exc:
            _record("U7", "get_osim_stats", False, str(exc))

    # -------------------------------------------------------------------
    # U8: tool_clamp_flexion_angle edge cases
    # -------------------------------------------------------------------
    if not _skip("U8"):
        _sub("U8 -- tool_clamp_flexion_angle")
        cases = [
            (30.0,   30.0,  "Normal angle unchanged"),
            (-10.0,   0.0,  "Negative clamped to 0"),
            (95.0,   90.0,  "Over-90 clamped to 90"),
            (0.0,     0.0,  "Zero stays zero"),
            (90.0,   90.0,  "90 stays 90"),
            (45.5,   45.5,  "Decimal unchanged"),
        ]
        for inp, expected, desc in cases:
            result = tool_clamp_flexion_angle(inp)
            _assert(abs(result - expected) < 1e-9, f"U8/{inp}",
                    f"clamp({inp}) = {result}  [{desc}]",
                    f"Expected {expected}, got {result}")

    # -------------------------------------------------------------------
    # U9: tool_validate_case
    # -------------------------------------------------------------------
    if not _skip("U9"):
        _sub("U9 -- tool_validate_case")
        try:
            good = {
                "label":              "test_case",
                "model_path":         _find_osim(),
                "motion_path":        "NMB_Motion2.mot",   # relative OK
                "force_path":         "NMB_ExternalForce2.mot",
                "perturbation_type":  "flexion_angle",
                "perturbation_value": 30.0,
            }
            missing_field = {k: v for k, v in good.items() if k != "label"}
            missing_file  = {**good, "model_path": "/nonexistent/path/model.osim"}

            v, _ = tool_validate_case(good)
            _assert(v,   "U9a", "Valid case passes validation")

            v, e = tool_validate_case(missing_field)
            _assert(not v, "U9b", f"Missing-field case fails validation: {e}")

            v, e = tool_validate_case(missing_file)
            _assert(not v, "U9c", f"Missing-file case fails validation: {e}")

        except Exception as exc:
            _record("U9", "tool_validate_case", False, str(exc))

    # -------------------------------------------------------------------
    # U10: Flexion variants with 2 kg hand load (20 deg +- 5 deg)
    # -------------------------------------------------------------------
    if not _skip("U10"):
        _sub("U10 -- generate_flexion_variants with 2 kg hand load")
        _u10_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u10")
        os.makedirs(_u10_tmp, exist_ok=True)
        try:
            from utils_sensitivity import generate_flexion_variants
            osim = _find_osim()

            cases = generate_flexion_variants(
                base_angle=20.0,
                delta_deg=5.0,
                weight_per_hand_kg=2.0,
                output_dir=_u10_tmp,
                model_path=osim,
                model_filename=os.path.basename(osim),
            )

            angles_got = sorted(c["perturbation_value"] for c in cases)
            _assert(len(cases) == 3, "U10a",
                    f"Flexion 20+-5 with 2kg: exactly 3 cases (got {len(cases)})")
            _assert(angles_got == [15.0, 20.0, 25.0], "U10b",
                    f"Angles are [15, 20, 25] deg (got {angles_got})")

            for c in cases:
                _assert(c.get("perturbation_type") == "flexion_angle", f"U10c/{c['perturbation_value']}",
                        f"perturbation_type = flexion_angle for {c['perturbation_value']}deg")
                _assert(os.path.exists(c["motion_path"]), f"U10d/{c['perturbation_value']}",
                        f"Motion file exists: {os.path.basename(c['motion_path'])}")
                _assert(os.path.exists(c["force_path"]),  f"U10e/{c['perturbation_value']}",
                        f"Force file exists: {os.path.basename(c['force_path'])}")

            # All cases share the same force file (same 2 kg load)
            force_paths = list(set(c["force_path"] for c in cases))
            _assert(len(force_paths) == 1, "U10f",
                    f"All 3 angles share the same force file (got {len(force_paths)} unique)")

        except Exception as exc:
            _record("U10", "generate_flexion_variants with load", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u10_tmp, ignore_errors=True)

    # -------------------------------------------------------------------
    # U11: Slack length +5% / -5% with neutral standing baseline
    # -------------------------------------------------------------------
    if not _skip("U11"):
        _sub("U11 -- _build_slack_cases: +-5% with neutral standing")
        _u11_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u11")
        os.makedirs(_u11_tmp, exist_ok=True)
        try:
            from utils_sensitivity import (
                _build_slack_cases, tool_get_base_motion_force_paths, DEFAULT_ACTIVITY_ID
            )
            osim     = _find_osim()
            paths    = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
            plan     = {"slack_length_pct": 5.0}

            cases = _build_slack_cases(
                plan=plan,
                model_path=osim,
                model_filename=os.path.basename(osim),
                temp_dir=_u11_tmp,
                activity_motion=paths["motion_path"],
                activity_force=paths["force_path"],
                activity_label="Neutral standing",
            )

            pert_vals = sorted(c["perturbation_value"] for c in cases)
            _assert(len(cases) == 3, "U11a",
                    f"Slack +-5%: exactly 3 cases (base, +5, -5) got {len(cases)}")
            _assert(-5.0 in pert_vals and 0.0 in pert_vals and 5.0 in pert_vals, "U11b",
                    f"Perturbation values are [-5, 0, +5] (got {pert_vals})")
            for c in cases:
                _assert(c.get("perturbation_type") == "slack_length_pct", f"U11c/{c['perturbation_value']}",
                        f"perturbation_type = slack_length_pct for val={c['perturbation_value']}")
                _assert(os.path.exists(c["model_path"]), f"U11d/{c['perturbation_value']}",
                        f"Modified model file exists on disk")

        except Exception as exc:
            _record("U11", "_build_slack_cases neutral standing", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u11_tmp, ignore_errors=True)

    # -------------------------------------------------------------------
    # U12: Vertebral +-2mm with 31-degree flexion baseline
    # -------------------------------------------------------------------
    if not _skip("U12"):
        _sub("U12 -- _build_vertebral_cases: +-2mm with 31-deg flexion baseline")
        _u12_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u12")
        os.makedirs(_u12_tmp, exist_ok=True)
        try:
            from utils_sensitivity import (
                _build_vertebral_cases, tool_clamp_flexion_angle,
                generate_interpolated_mot, generate_force_mot
            )
            osim      = _find_osim()
            angle     = tool_clamp_flexion_angle(31.0)
            mot_files = generate_interpolated_mot([angle], output_dir=_u12_tmp)
            frc_files = generate_force_mot(load_n=0.0,    output_dir=_u12_tmp)
            plan      = {"vertebral_delta_mm": 2.0}

            cases = _build_vertebral_cases(
                plan=plan,
                model_path=osim,
                model_filename=os.path.basename(osim),
                temp_dir=_u12_tmp,
                activity_motion=mot_files[0],
                activity_force=frc_files[0],
                activity_label="31deg trunk flexion",
            )

            pert_vals = sorted(c["perturbation_value"] for c in cases)
            _assert(len(cases) == 3, "U12a",
                    f"Vertebral +-2mm: exactly 3 cases (base, +2, -2) got {len(cases)}")
            _assert(-2.0 in pert_vals and 0.0 in pert_vals and 2.0 in pert_vals, "U12b",
                    f"Perturbation values are [-2, 0, +2] mm (got {pert_vals})")
            for c in cases:
                _assert(c.get("perturbation_type") == "vertebral_x_mm", f"U12c/{c['perturbation_value']}",
                        f"perturbation_type = vertebral_x_mm for val={c['perturbation_value']}")
                _assert(c["motion_path"] == mot_files[0], f"U12d/{c['perturbation_value']}",
                        "Activity is 31-deg flexion motion file")

        except Exception as exc:
            _record("U12", "_build_vertebral_cases 31-deg flexion", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u12_tmp, ignore_errors=True)

    # -------------------------------------------------------------------
    # U13: Gravity +-5% with upright standing baseline
    # -------------------------------------------------------------------
    if not _skip("U13"):
        _sub("U13 -- _build_gravity_cases: +-5% with neutral standing")
        _u13_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u13")
        os.makedirs(_u13_tmp, exist_ok=True)
        try:
            from utils_sensitivity import (
                _build_gravity_cases, tool_get_base_motion_force_paths, DEFAULT_ACTIVITY_ID
            )
            osim  = _find_osim()
            paths = tool_get_base_motion_force_paths(DEFAULT_ACTIVITY_ID)
            plan  = {"gravity_pct": 5.0}

            cases = _build_gravity_cases(
                plan=plan,
                model_path=osim,
                model_filename=os.path.basename(osim),
                temp_dir=_u13_tmp,
                activity_motion=paths["motion_path"],
                activity_force=paths["force_path"],
                activity_label="Neutral standing",
            )

            pert_vals = sorted(c["perturbation_value"] for c in cases)
            _assert(len(cases) == 3, "U13a",
                    f"Gravity +-5%: exactly 3 cases (base, +5, -5) got {len(cases)}")
            _assert(-5.0 in pert_vals and 0.0 in pert_vals and 5.0 in pert_vals, "U13b",
                    f"Perturbation values are [-5, 0, +5]% (got {pert_vals})")
            for c in cases:
                _assert(c.get("perturbation_type") == "gravity_pct", f"U13c/{c['perturbation_value']}",
                        f"perturbation_type = gravity_pct for val={c['perturbation_value']}")
                _assert(os.path.exists(c["model_path"]), f"U13d/{c['perturbation_value']}",
                        f"Modified model file exists on disk")

        except Exception as exc:
            _record("U13", "_build_gravity_cases neutral standing", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u13_tmp, ignore_errors=True)

    # -------------------------------------------------------------------
    # U14: Custom tool (PCSA) -- dynamic code generation and execution
    # -------------------------------------------------------------------
    if not _skip("U14"):
        _sub("U14 -- Custom tool 'pcsa_modification': dynamic code-gen and case building")
        _u14_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u14")
        os.makedirs(_u14_tmp, exist_ok=True)
        try:
            from utils_sensitivity import _build_all_sensitivity_cases, _KNOWN_TOOLS
            import io, contextlib

            osim = _find_osim()
            osim_rel = os.path.relpath(osim,
                                       os.path.join(PROJECT_DIR, "opensim_files"))

            # Minimal model info that mimics model_selection_node output
            fake_model = {
                "full_path": osim_rel.replace("\\", "/"),
                "Filename":  os.path.basename(osim),
            }

            plan_pcsa = {
                "tools": ["pcsa_modification"],
                "flexion_activities": [],
                "custom_tool_values": {
                    "pcsa_modification": 5.0
                },
                "user_prompt": "Do a sensitivity analysis on PCSA by 5 percent for upright standing"
            }

            cases   = None
            crashed = False
            try:
                cases = _build_all_sensitivity_cases(plan_pcsa, [fake_model])
            except Exception as exc:
                crashed = True
                _record("U14a", "Dynamic custom tool execution does not crash", False, str(exc))

            if not crashed:
                _assert(not crashed, "U14a",
                        "Pipeline did not crash when given dynamic custom tool 'pcsa_modification'")
                _assert(cases is not None and len(cases) == 3, "U14b",
                        f"Returns exactly 3 cases (base, +5, -5) for custom tool (got {len(cases) if cases else 'None'})")
                
                if cases and len(cases) == 3:
                    pert_vals = sorted(c["perturbation_value"] for c in cases)
                    _assert(-5.0 in pert_vals and 0.0 in pert_vals and 5.0 in pert_vals, "U14c",
                            f"Perturbation values are [-5, 0, 5]% (got {pert_vals})")
                    for c in cases:
                        _assert(c.get("perturbation_type") == "pcsa_modification", f"U14d/{c['perturbation_value']}",
                                f"perturbation_type = pcsa_modification for val={c['perturbation_value']}")
                        _assert(os.path.exists(c["model_path"]), f"U14e/{c['perturbation_value']}",
                                f"Generated model file exists: {c['model_path']}")

            _assert("pcsa_modification" not in _KNOWN_TOOLS, "U14f",
                    "pcsa_modification is NOT in _KNOWN_TOOLS (correct -- it is dynamic)")

        except Exception as exc:
            _record("U14", "Custom tool dynamic generation", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u14_tmp, ignore_errors=True)

    # -------------------------------------------------------------------
    # U15: Combinatorial -- 20deg +-5 x vertebral +-2mm x slack +-5% = 9 cases
    # -------------------------------------------------------------------
    if not _skip("U15"):
        _sub("U15 -- _build_combinatorial_cases: 2^3=8 combos + 1 baseline = 9 cases")
        _u15_tmp = os.path.join(PROJECT_DIR, "_test_tmp_u15")
        os.makedirs(_u15_tmp, exist_ok=True)
        try:
            from utils_sensitivity import _build_combinatorial_cases

            osim = _find_osim()

            plan_combo = {
                "combo_tools":          ["flexion_sweep", "slack_length", "vertebral_translation"],
                "combo_flexion_base":   20.0,
                "combo_flexion_delta":  5.0,
                "combo_slack_pct":      5.0,
                "combo_vertebral_mm":   2.0,
                "combo_weight_per_hand": 0.0,
            }

            cases = _build_combinatorial_cases(
                plan=plan_combo,
                model_path=osim,
                model_filename=os.path.basename(osim),
                temp_dir=_u15_tmp,
            )

            n_dims    = 3
            n_combos  = 2 ** n_dims   # 8
            expected  = n_combos + 1  # +1 baseline = 9

            _assert(len(cases) == expected, "U15a",
                    f"Total cases = {expected} (1 baseline + 2^3={n_combos} combos), got {len(cases)}")

            # Baseline check
            baseline = [c for c in cases if c.get("perturbation_value") == 0.0]
            _assert(len(baseline) == 1, "U15b",
                    f"Exactly 1 baseline case (got {len(baseline)})")

            # All cases have perturbation_type = "combinatorial"
            wrong_type = [c for c in cases if c.get("perturbation_type") != "combinatorial"]
            _assert(not wrong_type, "U15c",
                    f"All cases have perturbation_type='combinatorial' (wrong: {len(wrong_type)})")

            # All 8 combo cases have distinct combo_details
            combo_cases = [c for c in cases if isinstance(c.get("perturbation_value"), dict)]
            detail_sets = [str(sorted(c["combo_details"].items())) for c in combo_cases]
            _assert(len(set(detail_sets)) == len(combo_cases), "U15d",
                    f"All {len(combo_cases)} combo cases have unique combo_details")

            # All model files exist
            missing_models = [c for c in combo_cases if not os.path.exists(c["model_path"])]
            _assert(not missing_models, "U15e",
                    f"All combo model .osim files exist on disk ({len(missing_models)} missing)")

            # All motion files exist
            missing_motion = [c for c in cases if not os.path.exists(c["motion_path"])
                              and os.path.isabs(c["motion_path"])]
            _assert(not missing_motion, "U15f",
                    f"All motion .mot files exist ({len(missing_motion)} missing)")

            # Sign coverage: each dimension should appear with both +/- variants
            # For flexion_sweep, "negative" means angle < base; for others, val < 0 means minus.
            BASE_FLEXION = plan_combo["combo_flexion_base"]
            plus_minus = {"flexion_sweep": set(), "slack_length": set(), "vertebral_translation": set()}
            for c in combo_cases:
                for dim, val in c.get("combo_details", {}).items():
                    if dim in plus_minus:
                        if dim == "flexion_sweep":
                            sign = "pos" if float(val) > BASE_FLEXION else "neg"
                        else:
                            sign = "pos" if float(val) > 0 else "neg"
                        plus_minus[dim].add(sign)
            for dim, signs in plus_minus.items():
                _assert(len(signs) == 2, f"U15g/{dim[:6]}",
                        f"Dimension '{dim}' has both + and - variants (got {signs})")


            _info(f"Combo breakdown: {len(baseline)} baseline + {len(combo_cases)} combinations")

        except Exception as exc:
            _record("U15", "_build_combinatorial_cases", False, traceback.format_exc())
        finally:
            shutil.rmtree(_u15_tmp, ignore_errors=True)

    # Cleanup temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _info(f"Temp dir removed: {tmp_dir}")



# ===========================================================================
# INTEGRATION TEST DEFINITIONS
# ===========================================================================

INTEGRATION_TESTS = [
    {
        "id": "I1",
        "name": "Flexion sweep -- Tool 1: interpolation +-5 deg",
        "prompt": (
            "Sensitivity analysis on 30 degree trunk flexion with plus minus 5 degrees "
            "for a 70 kg male 1.72 m tall"
        ),
        "expected_tools":         ["flexion_sweep"],
        "expected_pert_types":    ["flexion_angle"],
        "expect_min_success_runs": 1,
    },
    {
        "id": "I2",
        "name": "Slack length -- Tool 2: XML +-5 percent",
        "prompt": (
            "Do a sensitivity analysis on muscle passive properties slack length "
            "plus minus 5 percent for a 70 kg male 1.72 m standing"
        ),
        "expected_tools":         ["slack_length"],
        "expected_pert_types":    ["slack_length_pct"],
        "expect_min_success_runs": 1,
    },
    {
        "id": "I3",
        "name": "Vertebral translation -- Tool 3: XML +-2 mm",
        "prompt": (
            "Sensitivity analysis on vertebral joint positions anterior posterior "
            "direction plus minus 2 mm for a 70 kg male 1.72 m tall standing"
        ),
        "expected_tools":         ["vertebral_translation"],
        "expected_pert_types":    ["vertebral_x_mm"],
        "expect_min_success_runs": 1,
    },
    {
        "id": "I4",
        "name": "Gravity scaling -- Tool 4: XML +-10 percent",
        "prompt": (
            "Sensitivity analysis on gravitational load plus minus 10 percent "
            "for a 70 kg male 1.72 m tall neutral standing"
        ),
        "expected_tools":         ["gravity"],
        "expected_pert_types":    ["gravity_pct"],
        "expect_min_success_runs": 1,
    },
    {
        "id": "I5",
        "name": "FF-mode bare -- all 4 tools, default values",
        "prompt": "ff-mode for a 70 kg male 1.72 m tall",
        "expected_tools":         ["flexion_sweep", "slack_length",
                                   "vertebral_translation", "gravity"],
        "expected_pert_types":    ["flexion_angle", "slack_length_pct",
                                   "vertebral_x_mm", "gravity_pct"],
        "expect_min_success_runs": 2,
    },
    {
        "id": "I6",
        "name": "Flexion + hand load -- extra case: 20 deg +-5 deg, 5 kg/hand",
        "prompt": (
            "Sensitivity analysis on 20 degree trunk flexion plus minus 5 degrees "
            "with 5 kg weight in each hand for a 70 kg male 1.72 m"
        ),
        "expected_tools":         ["flexion_sweep"],
        "expected_pert_types":    ["flexion_angle"],
        "expect_min_success_runs": 1,
    },
]


# ===========================================================================
# INTEGRATION TEST RUNNER
# ===========================================================================

def run_integration_test(tdef: dict):
    tid    = tdef["id"]
    name   = tdef["name"]
    prompt = tdef["prompt"]

    _hdr(f"INTEGRATION {tid} -- {name}")
    _info(f"Prompt: \"{prompt}\"")

    t0 = time.time()

    # Keyword detection check (cheap)
    from utils_sensitivity import is_sensitivity_prompt
    is_sens = is_sensitivity_prompt(prompt)
    if not _assert(is_sens, f"{tid}/kw",
                   f"Keyword detection: is_sensitivity_prompt = {is_sens}"):
        _warn("Prompt not detected as sensitivity -- skipping pipeline run")
        return

    # Full pipeline
    _info("Invoking app.invoke() -- this may take several minutes ...")
    try:
        from main import app
        result = app.invoke({
            "user_prompt":         prompt,
            "uploaded_image_path": None,
        })
    except Exception:
        _record(f"{tid}/run", "app.invoke() completed without exception",
                False, traceback.format_exc())
        return

    elapsed = time.time() - t0
    _info(f"Pipeline finished in {elapsed:.1f} s")

    # final_message
    final = result.get("final_message", "")
    _assert(final and len(final) > 20, f"{tid}/msg",
            f"final_message present ({len(final)} chars)")

    # Sensitivity plan
    plan = result.get("sensitivity_plan") or {}
    if plan:
        tools_used = plan.get("tools", [])
        _info(f"Plan tools: {tools_used}")
        for et in tdef.get("expected_tools", []):
            _assert(et in tools_used, f"{tid}/tool/{et}",
                    f"Tool '{et}' in plan", f"Actual: {tools_used}")
    else:
        _warn(f"[{tid}] No sensitivity_plan in result")

    # Simulation results
    sens_res = result.get("sensitivity_results") or []
    if sens_res:
        success  = sum(1 for r in sens_res if r.get("status") == "Success")
        total    = len(sens_res)
        pert_types = list(set(r.get("perturbation_type", "") for r in sens_res))
        _info(f"Simulations: {success}/{total} succeeded")
        _info(f"Perturbation types in results: {pert_types}")

        _assert(total > 0, f"{tid}/cases",
                f"At least one case was generated ({total} total)")
        min_ok = tdef.get("expect_min_success_runs", 1)
        _assert(success >= min_ok, f"{tid}/success",
                f"At least {min_ok} simulation(s) succeeded ({success}/{total})",
                "Check opensim_files/sensitivity_temp/ for generated files")

        for ep in tdef.get("expected_pert_types", []):
            _assert(ep in pert_types, f"{tid}/ptype/{ep}",
                    f"Perturbation type '{ep}' present", f"Actual: {pert_types}")
    else:
        _record(f"{tid}/cases", "sensitivity_results present in state", False,
                "result['sensitivity_results'] is empty or missing")

    # DataFrames
    dfs = result.get("dataframes") or {}
    for dfname in ("spinal", "forces", "activations"):
        df = dfs.get(dfname)
        _assert(df is not None and not df.empty, f"{tid}/df/{dfname}/present",
                f"DataFrame '{dfname}' is present and not empty")
        if df is not None and not df.empty:
            has_col = "Perturbation_Type" in df.columns
            _assert(has_col, f"{tid}/df/{dfname}/cols",
                    f"DataFrame '{dfname}' has Perturbation_Type column ({len(df)} rows)",
                    f"Columns: {list(df.columns)}")
            if has_col:
                _info(f"  '{dfname}': {len(df)} rows | "
                      f"types = {df['Perturbation_Type'].unique().tolist()}")

    # Reflections / errors
    errors = result.get("sensitivity_errors") or []
    iters  = result.get("sensitivity_iteration", 0)
    if errors:
        _warn(f"{iters} reflection(s) used; {len(errors)} error-log entries")
        for e in errors[:3]:
            _info(f"  {e.get('original','?')} -> {e.get('fix','?')} "
                  f"[{e.get('retry_status','?')}]")

    # Preview
    preview = final.replace("\n", " ")[:280]
    _info(f"Final msg preview: {preview}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Sensitivity pipeline test suite")
    parser.add_argument("--unit",  action="store_true",
                        help="Unit tests only (fast, no API/OpenSim)")
    parser.add_argument("--integ", action="store_true",
                        help="Integration tests only")
    parser.add_argument("--id",    type=str,
                        help="Run single test by ID, e.g. --id I1 or --id U3")
    args = parser.parse_args()

    run_unit  = (not args.integ) or (args.id and args.id.startswith("U"))
    run_integ = (not args.unit)  or (args.id and args.id.startswith("I"))

    # If --id is given, restrict accordingly
    if args.id:
        run_unit  = args.id.startswith("U")
        run_integ = args.id.startswith("I")

    start = time.time()

    if run_unit:
        try:
            run_unit_tests(filter_id=args.id if args.id and args.id.startswith("U") else None)
        except Exception:
            print("[ERROR] Unit test suite crashed:")
            traceback.print_exc()

    if run_integ:
        tests = INTEGRATION_TESTS
        if args.id:
            tests = [t for t in INTEGRATION_TESTS if t["id"] == args.id]
            if not tests:
                print(f"\n[ERROR] No integration test with id '{args.id}'")

        for tdef in tests:
            try:
                run_integration_test(tdef)
            except Exception:
                _record(tdef["id"], tdef["name"], False, traceback.format_exc())

    # Summary
    total   = len(RESULTS)
    passed  = sum(1 for _, _, p, _ in RESULTS if p)
    failed  = total - passed
    elapsed = time.time() - start

    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  TEST SUMMARY   ({elapsed:.1f} s total)")
    print(bar)
    print(f"  Passed: {passed}  /  Failed: {failed}  /  Total: {total}")

    if failed:
        print("\n  FAILED TESTS:")
        for tid, name, p, notes in RESULTS:
            if not p:
                print(f"    [FAIL] [{tid}] {name}")
                if notes:
                    short = str(notes)[:120].replace("\n", " ")
                    print(f"           {short}")

    print(bar)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
