"""
generate_force_mot.py
---------------------
Generates OpenSim external-force .mot files with a user-supplied hand load.

The template file (NMB_ExternalForce28.mot) contains a constant vertical
hand force of -49.05 N in columns Hand_R_Force_y and Hand_L_Force_y.
This function replaces that value with any load you pass in and writes a
new .mot file with a meaningful name.

Column layout (0-indexed):
    0  : time
    1  : Hand_R_Force_x
    2  : Hand_R_Force_y   ← replaced by `load_n`
    3  : Hand_R_Force_z
    4  : Hand_R_pt_x
    5  : Hand_R_pt_y
    6  : Hand_R_pt_z
    7  : Hand_L_Force_x
    8  : Hand_L_Force_y   ← replaced by `load_n`
    9  : Hand_L_Force_z
    10 : Hand_L_pt_x
    11 : Hand_L_pt_y
    12 : Hand_L_pt_z
    13-24 : scapula forces / points (unchanged)

Usage
-----
    from generate_force_mot import generate_force_mot

    # Single load
    generate_force_mot(load_n=-98.1, output_dir=r"C:/path/to/output")

    # Multiple loads at once
    generate_force_mot(
        load_n=[-49.05, -98.1, -196.2],
        output_dir=r"C:/path/to/output",
    )
"""

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Template file — structure is preserved; only the hand-force columns change.
TEMPLATE_FILE = os.path.join(
    os.path.dirname(__file__), "flexion force", "NMB_ExternalForce28.mot"
)

# Column indices whose value is replaced by the user-supplied load
FORCE_Y_COLS = {2, 8}   # Hand_R_Force_y, Hand_L_Force_y


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_force_mot(filepath):
    """
    Parse the template .mot file.

    Returns
    -------
    header_lines : list[str]
        Lines 1-7 (indices 0-6), raw text (including the column-name row).
    data_rows : list[list[str]]
        Lines 8-18 split into string tokens (preserves exact formatting of
        all unchanged columns).
    """
    with open(filepath, "r") as f:
        raw = f.readlines()

    header_lines = raw[:7]                              # lines 1-7
    data_lines   = [l for l in raw[7:] if l.strip()]   # skip blank trailing lines
    data_rows    = [l.split() for l in data_lines]
    return header_lines, data_rows


def _write_force_mot(output_path, mot_name, header_lines, data_rows, load_n):
    """
    Write a new .mot file replacing FORCE_Y_COLS with `load_n`.

    Parameters
    ----------
    output_path : str
        Full path for the output file.
    mot_name : str
        Name embedded on line 1 of the file.
    header_lines : list[str]
        Original header from the template (7 lines).
    data_rows : list[list[str]]
        Tokenised data rows from the template.
    load_n : float
        Hand load in Newtons (negative = downward in OpenSim convention).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="\r\n") as f:
        # Line 1: motion / force file name
        f.write(mot_name + "\n")

        # Lines 2-7: rest of the header unchanged
        for line in header_lines[1:]:
            f.write(line.rstrip("\r\n") + "\n")

        # Data rows: swap force-y columns
        for row in data_rows:
            new_row = []
            for col_idx, token in enumerate(row):
                if col_idx in FORCE_Y_COLS:
                    new_row.append(f"{load_n:18.8f}")
                else:
                    # Re-format as float to keep consistent column width
                    new_row.append(f"{float(token):18.8f}")
            f.write("\t".join(new_row) + "\n")

        f.write("\n")   # trailing blank line (matches original format)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_force_mot(load_n, output_dir=None, name_prefix="NMB_ExternalForce_load"):
    """
    Generate one or more OpenSim external-force .mot files.

    Parameters
    ----------
    load_n : float or list[float]
        Hand load(s) in Newtons applied to Hand_R_Force_y and Hand_L_Force_y.
        Use negative values for downward forces (OpenSim convention),
        e.g. -49.05 for a 5 kg hand load (5 × 9.81).
    output_dir : str or None
        Directory where output files are written.
        Defaults to the directory of this script.
    name_prefix : str
        Prefix for output file names.
        Files are named  <name_prefix><abs_load>N.mot

    Returns
    -------
    list[str]
        Absolute paths of all files written.
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    # Accept a single value or a list
    if not isinstance(load_n, (list, tuple)):
        load_n = [load_n]

    if not os.path.isfile(TEMPLATE_FILE):
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_FILE}")

    header_lines, data_rows = _parse_force_mot(TEMPLATE_FILE)

    written = []
    for load in load_n:
        # Build a readable file name from the load magnitude
        abs_load  = abs(load)
        load_tag  = f"{abs_load:.2f}".replace(".", "p")   # e.g. 49p05
        mot_name  = f"{name_prefix}{load_tag}N"
        filename  = f"{mot_name}.mot"
        out_path  = os.path.join(output_dir, filename)

        _write_force_mot(out_path, mot_name, header_lines, data_rows, load)
        written.append(out_path)
        print(f"  [OK] {filename}  (Hand_Force_y = {load:.4f} N)")

    print(f"\nDone — {len(written)} file(s) written to:\n  {os.path.abspath(output_dir)}")
    return written


# ---------------------------------------------------------------------------
# Entry point (run directly for a quick demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

    demo_loads = [-49.05, -98.1, -196.2]   # 5 kg, 10 kg, 20 kg hand loads

    print("Generating force .mot files ...")
    generate_force_mot(
        load_n=demo_loads,
        output_dir=OUTPUT_DIR,
    )
