"""
match_mri_to_model.py
=====================
Given disc centroids from segment_spine_mri.py, compute discrete inter-vertebral
angles and find the closest OpenSim musculoskeletal model(s) from
spine_curvature_metrics.csv using a KNN-style distance that gracefully handles
missing (partially-visible) spinal levels.

Flip handling
-------------
Because the MRI pixel axis may be flipped relative to the OpenSim anatomical
convention, matching is run TWICE:
  Pass 1 — original angles
  Pass 2 — all angles × -1  (simulates a horizontally-flipped image)
The globally best top-k results across both passes are returned, each tagged
with "original" or "flipped" to indicate which pass produced that match.

Importable API
--------------
    from opensim_files.match_mri_to_model import run_matching
    matches = run_matching(centroids, k=5)

Standalone
----------
    python match_mri_to_model.py
    (uses the last disc-centroid CSV found in results_medical_image_segmentation/)
"""

import os
import math
import csv
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this file's location = opensim_files/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

DEFAULT_METRICS_CSV = os.path.join(_HERE, "spine_curvature_metrics.csv")
DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results_medical_image_segmentation")

# ---------------------------------------------------------------------------
# Mapping: MRI disc name → OpenSim CSV column name for the discrete angle
#
# The discrete angle stored in the CSV for "lumbar4" is the signed angle
# between the segment (lumbar5→lumbar4) and (lumbar4→lumbar3), which
# geometrically corresponds to the curvature *at* the L4-L5 disc space.
# ---------------------------------------------------------------------------
_DISC_NAME_TO_CSV_COL: Dict[str, str] = {
    "L4-L5":   "discrete_angle_lumbar4_deg",
    "L3-L4":   "discrete_angle_lumbar3_deg",
    "L2-L3":   "discrete_angle_lumbar2_deg",
    "L1-L2":   "discrete_angle_lumbar1_deg",
    "T12-L1":  "discrete_angle_thoracic12_deg",
    "T11-T12": "discrete_angle_thoracic11_deg",
    "T10-T11": "discrete_angle_thoracic10_deg",
    "T9-T10":  "discrete_angle_thoracic9_deg",
    "T8-T9":   "discrete_angle_thoracic8_deg",
    "T7-T8":   "discrete_angle_thoracic7_deg",
    "T6-T7":   "discrete_angle_thoracic6_deg",
    "T5-T6":   "discrete_angle_thoracic5_deg",
    "T4-T5":   "discrete_angle_thoracic4_deg",
    "T3-T4":   "discrete_angle_thoracic3_deg",
    "T2-T3":   "discrete_angle_thoracic2_deg",
}

# The disc names in the correct superior-to-inferior order (T2-T3 is highest)
_DISC_ORDER: List[str] = [
    "T2-T3", "T3-T4", "T4-T5", "T5-T6", "T6-T7", "T7-T8",
    "T8-T9", "T9-T10", "T10-T11", "T11-T12", "T12-L1",
    "L1-L2", "L2-L3", "L3-L4", "L4-L5",
]

# Aliases produced by TotalSegmentator that we normalise to our canonical names
_DISC_ALIASES: Dict[str, str] = {
    # TotalSegmentator sometimes uses different separator or capitalisation
    "L4_L5":   "L4-L5",
    "L3_L4":   "L3-L4",
    "L2_L3":   "L2-L3",
    "L1_L2":   "L1-L2",
    "T12_L1":  "T12-L1",
    "T11_T12": "T11-T12",
    "T10_T11": "T10-T11",
    "T9_T10":  "T9-T10",
    "T8_T9":   "T8-T9",
    "T7_T8":   "T7-T8",
    "T6_T7":   "T6-T7",
    "T5_T6":   "T5-T6",
    "T4_T5":   "T4-T5",
    "T3_T4":   "T3-T4",
    "T2_T3":   "T2-T3",
}


# ---------------------------------------------------------------------------
# Step 1: Parse MRI disc centroids
# ---------------------------------------------------------------------------

def parse_mri_disc_centroids(
    centroids: List[dict],
) -> List[Tuple[str, float, float]]:
    """
    Extract (disc_name, col_px, row_px) for known thoracic/lumbar discs,
    sorted from superior to inferior (i.e., T2-T3 first, L4-L5 last).

    Parameters
    ----------
    centroids : list of dicts from compute_disc_centroids()
        Each dict has keys: "name", "c2_row", "c2_col" (2-D mid-slice position).

    Returns
    -------
    List of (canonical_disc_name, col_px, row_px) tuples.
    Only discs that have a 2-D centroid (c2_row is not None) are included.
    """
    known: Dict[str, Tuple[float, float]] = {}

    for c in centroids:
        raw_name: str = c["name"]
        # Normalise separator
        canonical = raw_name.replace("_", "-").strip()
        canonical = _DISC_ALIASES.get(canonical, canonical)

        if canonical not in _DISC_NAME_TO_CSV_COL:
            continue  # cervical, sacral, or unrecognised disc — skip
        if c["c2_row"] is None or c["c2_col"] is None:
            continue  # disc not visible in the mid-slice

        known[canonical] = (float(c["c2_col"]), float(c["c2_row"]))

    # Sort into anatomical order
    ordered = [
        (name, *known[name])
        for name in _DISC_ORDER
        if name in known
    ]
    return ordered  # list of (name, col, row)


# ---------------------------------------------------------------------------
# Step 2: Compute discrete inter-vertebral angles from MRI pixel positions
# ---------------------------------------------------------------------------

def compute_mri_discrete_angles(
    disc_positions: List[Tuple[str, float, float]],
) -> Dict[str, float]:
    """
    Compute a discrete angle for each interior disc (all except the two end
    discs that form the boundary).

    The angle at disc i is the signed angle between:
        vector u = P_{i-1} → P_i
        vector w = P_i     → P_{i+1}

    This is identical in concept to the segmental angle calculation in
    plot_spine_coords.py (lines 189-208), adapted for 2-D pixel space.

    The sign convention follows atan2(cross, dot):
        positive = counter-clockwise change (lordotic direction)
        negative = clockwise change (kyphotic direction)

    Returns
    -------
    Dict mapping disc_name → angle_degrees.
    Discs at the top/bottom of the visible range get no angle (need neighbours
    on both sides), so they are omitted.
    """
    if len(disc_positions) < 3:
        return {}

    angles: Dict[str, float] = {}

    for i in range(1, len(disc_positions) - 1):
        name_prev, col_prev, row_prev = disc_positions[i - 1]
        name_curr, col_curr, row_curr = disc_positions[i]
        name_next, col_next, row_next = disc_positions[i + 1]

        # Segment vector u: previous → current
        ux = col_curr - col_prev
        uy = row_curr - row_prev

        # Segment vector w: current → next
        wx = col_next - col_curr
        wy = row_next - row_curr

        dot   = ux * wx + uy * wy
        cross = ux * wy - uy * wx

        if abs(dot) < 1e-9 and abs(cross) < 1e-9:
            continue  # degenerate (zero-length segment)

        angle_deg = math.degrees(math.atan2(cross, dot))
        angles[name_curr] = angle_deg

    return angles


# ---------------------------------------------------------------------------
# Step 3: Load OpenSim model angles from spine_curvature_metrics.csv
# ---------------------------------------------------------------------------

def load_opensim_angles(
    csv_path: str = DEFAULT_METRICS_CSV,
) -> List[dict]:
    """
    Read spine_curvature_metrics.csv and return a list of per-model dicts.

    Each dict has:
        "model_file"   : str
        "sex"          : str
        "age_group"    : str
        "lumbar_lordosis_deg"    : float
        "thoracic_kyphosis_deg"  : float
        "disc_angles"  : Dict[disc_name → float]
            keys are canonical disc names (e.g. "L4-L5")
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"spine_curvature_metrics.csv not found at: {csv_path}\n"
            "Run plot_spine_coords.py first to generate it."
        )

    df = pd.read_csv(csv_path)
    models = []

    # Build reverse mapping: csv_col → disc_name
    col_to_disc = {v: k for k, v in _DISC_NAME_TO_CSV_COL.items()}

    for _, row in df.iterrows():
        disc_angles: Dict[str, float] = {}
        for col_name, disc_name in col_to_disc.items():
            if col_name in row and not pd.isna(row[col_name]):
                disc_angles[disc_name] = float(row[col_name])

        models.append({
            "model_file":             str(row.get("model_file", "")),
            "sex":                    str(row.get("sex", "")),
            "age_group":              str(row.get("age_group", "")),
            "lumbar_lordosis_deg":    float(row["lumbar_lordosis_deg"])
                                      if "lumbar_lordosis_deg" in row else None,
            "thoracic_kyphosis_deg":  float(row["thoracic_kyphosis_deg"])
                                      if "thoracic_kyphosis_deg" in row else None,
            "disc_angles":            disc_angles,
        })

    return models


# ---------------------------------------------------------------------------
# Step 4: KNN matching — partial levels handled naturally
# ---------------------------------------------------------------------------

def find_closest_models(
    mri_angles: Dict[str, float],
    all_models: List[dict],
    k: int = 5,
    min_shared_levels: int = 2,
) -> List[dict]:
    """
    Rank all OpenSim models by how closely their discrete segmental angles
    match the MRI angles, using only the levels visible in both.

    Distance metric
    ---------------
    mean( |mri_angle[level] - model_angle[level]| )  over shared levels.
    We use absolute differences (magnitude) to be robust to axis-sign
    ambiguity between MRI pixel space and OpenSim anatomical space.

    Parameters
    ----------
    mri_angles : dict  disc_name → angle_deg   (from compute_mri_discrete_angles)
    all_models : list  (from load_opensim_angles)
    k          : int   number of top matches to return
    min_shared_levels : int  minimum shared levels to include a model

    Returns
    -------
    List of result dicts (up to k), sorted by ascending distance:
        "rank", "distance_deg", "n_shared_levels",
        "model_file", "sex", "age_group",
        "lumbar_lordosis_deg", "thoracic_kyphosis_deg"
    """
    results = []

    for model in all_models:
        model_angles = model["disc_angles"]
        shared = [lvl for lvl in mri_angles if lvl in model_angles]

        if len(shared) < min_shared_levels:
            continue

        abs_diffs = [abs(mri_angles[lvl] - model_angles[lvl]) for lvl in shared]
        distance = float(np.mean(abs_diffs))

        results.append({
            "model_file":            model["model_file"],
            "sex":                   model["sex"],
            "age_group":             model["age_group"],
            "lumbar_lordosis_deg":   model["lumbar_lordosis_deg"],
            "thoracic_kyphosis_deg": model["thoracic_kyphosis_deg"],
            "distance_deg":          round(distance, 4),
            "n_shared_levels":       len(shared),
            "shared_levels":         shared,
        })

    results.sort(key=lambda r: r["distance_deg"])

    for rank, r in enumerate(results[:k], start=1):
        r["rank"] = rank

    return results[:k]


# ---------------------------------------------------------------------------
# Step 5: Top-level convenience function  (two-pass flip-aware)
# ---------------------------------------------------------------------------

def run_matching(
    centroids: List[dict],
    metrics_csv: str = DEFAULT_METRICS_CSV,
    k: int = 5,
    verbose: bool = True,
) -> List[dict]:
    """
    Full pipeline: centroids → angles → two-pass KNN match → top-k results.

    Two passes are run to handle MRI axis-flip ambiguity:
      Pass 1 — angles as computed from pixel coordinates
      Pass 2 — all angles multiplied by -1 (equivalent to a flipped image)

    Results from both passes are merged; the globally best top-k unique models
    are returned, each tagged with 'orientation': 'original' or 'flipped'.

    Parameters
    ----------
    centroids   : list of dicts from compute_disc_centroids()
    metrics_csv : path to spine_curvature_metrics.csv
    k           : number of top models to return
    verbose     : print results to stdout

    Returns
    -------
    List of match dicts, each with extra key 'orientation' ('original'/'flipped').
    """
    print("\n=== MRI → OpenSim Model Matching (flip-aware, 2 passes) ===")

    # ── 1. Parse disc centroids ───────────────────────────────────────────────
    disc_positions = parse_mri_disc_centroids(centroids)
    if len(disc_positions) == 0:
        print("  WARNING: No known thoracic/lumbar disc centroids found in MRI. "
              "Cannot perform matching.")
        return []

    print(f"  Visible disc levels in MRI ({len(disc_positions)}): "
          f"{[d[0] for d in disc_positions]}")

    # ── 2. Compute MRI discrete angles ────────────────────────────────────────
    mri_angles_orig = compute_mri_discrete_angles(disc_positions)
    if len(mri_angles_orig) == 0:
        print("  WARNING: Need at least 3 disc levels to compute angles. "
              "Cannot perform matching.")
        return []

    # Pass 2: flip all angles by ×(-1)
    mri_angles_flip = {lvl: -ang for lvl, ang in mri_angles_orig.items()}

    print(f"  Angles computed for {len(mri_angles_orig)} interior levels:")
    print(f"  {'Level':>10}  {'Original':>10}  {'Flipped':>10}")
    print("  " + "-" * 36)
    for lvl in mri_angles_orig:
        print(f"  {lvl:>10}  {mri_angles_orig[lvl]:>+9.2f}°  "
              f"{mri_angles_flip[lvl]:>+9.2f}°")

    # ── 3. Load OpenSim model angles ──────────────────────────────────────────
    all_models = load_opensim_angles(metrics_csv)
    print(f"\n  Loaded {len(all_models)} OpenSim models from: {metrics_csv}")

    # ── 4. KNN matching — both passes, fetch 2k candidates each ───────────────
    # We fetch 2k per pass so that after merging we still have k unique models.
    matches_orig = find_closest_models(mri_angles_orig, all_models, k=k * 2)
    matches_flip = find_closest_models(mri_angles_flip, all_models, k=k * 2)

    # Tag each result with its orientation
    for m in matches_orig:
        m["orientation"] = "original"
    for m in matches_flip:
        m["orientation"] = "flipped"

    # ── 5. Merge: keep best distance per unique model across both passes ───────
    best_per_model: Dict[str, dict] = {}
    for m in matches_orig + matches_flip:
        key = m["model_file"]
        if key not in best_per_model or m["distance_deg"] < best_per_model[key]["distance_deg"]:
            best_per_model[key] = m

    merged = sorted(best_per_model.values(), key=lambda r: r["distance_deg"])[:k]

    if not merged:
        print("  WARNING: No models had sufficient shared levels for matching.")
        return []

    # Re-assign ranks on the merged list
    for rank, m in enumerate(merged, start=1):
        m["rank"] = rank

    # ── 6. Print summary ──────────────────────────────────────────────────────
    # Quick stats: how many unique models came from each pass
    n_orig = sum(1 for m in merged if m["orientation"] == "original")
    n_flip = sum(1 for m in merged if m["orientation"] == "flipped")
    print(f"\n  Pass summary: {n_orig} best match(es) from original, "
          f"{n_flip} from flipped orientation.")

    print(f"\n  Top-{k} closest OpenSim models (merged across both passes):\n")
    print(f"  {'Rank':>4}  {'Distance':>10}  {'Shared':>6}  {'Orient':>8}  "
          f"{'Sex':>6}  {'Age':>10}  {'LL°':>7}  {'TK°':>7}  Model")
    print("  " + "-" * 95)
    for m in merged:
        ll = f"{m['lumbar_lordosis_deg']:.1f}" if m["lumbar_lordosis_deg"] else "N/A"
        tk = f"{m['thoracic_kyphosis_deg']:.1f}" if m["thoracic_kyphosis_deg"] else "N/A"
        print(
            f"  {m['rank']:>4}  {m['distance_deg']:>9.2f}°  "
            f"{m['n_shared_levels']:>6}  {m['orientation']:>8}  "
            f"{m['sex']:>6}  {m['age_group']:>10}  "
            f"{ll:>7}  {tk:>7}  {m['model_file']}"
        )

    best = merged[0]
    print(
        f"\n  ✔ Best match : {best['model_file']}  "
        f"({best['sex']}, {best['age_group']})\n"
        f"    distance   : {best['distance_deg']:.2f}° over "
        f"{best['n_shared_levels']} shared levels\n"
        f"    orientation: {best['orientation']}"
    )

    return merged


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _load_centroids_from_csv(csv_path: str) -> List[dict]:
    """Load a disc-centroids CSV saved by save_centroids_csv() into a list
    of dicts compatible with compute_disc_centroids() output format."""
    centroids = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                c2_row = float(row["row"]) if row["row"] != "N/A" else None
                c2_col = float(row["col"]) if row["col"] != "N/A" else None
            except (ValueError, KeyError):
                c2_row = c2_col = None
            centroids.append({
                "name":   row["disc"],
                "c2_row": c2_row,
                "c2_col": c2_col,
            })
    return centroids


if __name__ == "__main__":
    import sys

    # Allow passing a specific centroid CSV as argument
    if len(sys.argv) >= 2:
        centroid_csv = sys.argv[1]
    else:
        # Auto-detect: pick the most recently modified disc_centroids CSV
        pattern = os.path.join(DEFAULT_RESULTS_DIR, "*_disc_centroids_*.csv")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            print(
                f"No disc centroid CSV found in {DEFAULT_RESULTS_DIR}.\n"
                "Run segment_spine_mri.py first, or pass the CSV path as argument."
            )
            sys.exit(1)
        centroid_csv = candidates[0]

    print(f"Loading centroids from: {centroid_csv}")
    centroids = _load_centroids_from_csv(centroid_csv)
    matches = run_matching(centroids, k=5)
