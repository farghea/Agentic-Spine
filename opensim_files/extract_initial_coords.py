#%%
# ============================================================
#  extract_initial_coords.py  –  TEMPORARY / STANDALONE
#  Loads every .osim model in Male/ and Female/ (all age
#  sub-folders), calls initSystem() (= upright standing),
#  reads the 3D POSITION (x, y, z) of each vertebral body
#  in the global ground frame, and writes all vertebrae for
#  one model in a SINGLE ROW to:
#      opensim_files/initial_coords_all_models.csv
# ============================================================

import os
from glob import glob

import opensim as osim
import pandas as pd

# ── 1. Locate all .osim files ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # opensim_files/

osim_files = sorted(
    glob(os.path.join(BASE_DIR, "Male",   "**", "*.osim"), recursive=True) +
    glob(os.path.join(BASE_DIR, "Female", "**", "*.osim"), recursive=True)
)

total_models = len(osim_files)
print(f"Found {total_models} model(s).\n")

# ── 2. Initialize CSV (will be created on first write) ───────
out_path = os.path.join(BASE_DIR, "initial_coords_all_models.csv")
header_written = False

# ── 3. Extract vertebra body positions for each model ────────
for idx, model_path in enumerate(osim_files, start=1):
    rel   = os.path.relpath(model_path, BASE_DIR)
    parts = rel.replace("\\", "/").split("/")
    sex   = parts[0]                              # Male / Female
    age   = parts[1] if len(parts) > 2 else ""   # Age5059 etc.
    fname = os.path.basename(model_path)

    pct = (idx / total_models) * 100
    print(f"[{idx}/{total_models}] ({pct:.1f}%)  {rel} ...", end=" ", flush=True)

    try:
        model = osim.Model(model_path)
        state = model.initSystem()   # upright standing; state needed for position queries

        # Build ONE flat row: 3 meta columns + x/y/z per vertebral body
        row = {"sex": sex, "age_group": age, "model_file": fname}

        body_set = model.getBodySet()
        for i in range(body_set.getSize()):
            body       = body_set.get(i)
            body_name  = body.getName()
            body_lower = body_name.lower()

            # Keep only vertebral bodies (lumbar* and thoracic* bodies)
            is_vertebra = (
                "lumbar"   in body_lower or
                "thoracic" in body_lower or
                "vert"     in body_lower
            )
            if not is_vertebra:
                continue

            # 3-D position of this body's origin in the global ground frame (meters)
            pos = body.getPositionInGround(state)
            row[f"{body_name}_x"] = round(pos[0], 6)
            row[f"{body_name}_y"] = round(pos[1], 6)
            row[f"{body_name}_z"] = round(pos[2], 6)

        # Append one row to CSV (write header only on the first model)
        df_row = pd.DataFrame([row])
        df_row.to_csv(out_path, mode='a', header=not header_written, index=False)
        header_written = True

        n_bodies = (len(row) - 3) // 3   # subtract 3 meta cols; 3 values per body
        print(f"OK  ({n_bodies} vertebrae)")

    except Exception as e:
        print(f"FAILED – {e}")

print(f"\nDone. Results in:\n  {out_path}")
