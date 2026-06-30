#%%

import io
import json
import os
import shutil
import sys
from glob import glob

import opensim as osim
import pandas as pd


BASE_OUTPUT_KEYS = ["spinal_loads", "muscle_forces", "muscle_activations"]


def log(msg):
    # Keep logs on stderr so stdout remains valid JSON.
    sys.stderr.write(msg + "\n")


def read_sto(filename):
    try:
        with open(filename, "r") as f:
            while f.readline().strip() != "endheader":
                pass
            columns = f.readline().split()
            for line in f:
                values = line.split()
                if values and abs(float(values[0])) < 1e-6:
                    return dict(zip(columns, values))
            return "no results are written"
    except Exception:
        return "no results are written"


def apply_external_force_config(loads_file, config, results_dir):
    hand_cfg = config.get("hand_load", {}) if isinstance(config, dict) else {}
    shift_cfg = config.get("force_point_shift_m", {}) if isinstance(config, dict) else {}

    mode = hand_cfg.get("mode", "none")
    kg_each = hand_cfg.get("kg_each")
    scale = hand_cfg.get("scale")
    shift_x = float(shift_cfg.get("x", 0.0))
    shift_y = float(shift_cfg.get("y", 0.0))
    shift_z = float(shift_cfg.get("z", 0.0))

    needs_mod = (
        mode in {"set_each_hand_kg", "scale_existing"}
        or abs(shift_x) > 0
        or abs(shift_y) > 0
        or abs(shift_z) > 0
    )
    if not needs_mod:
        return loads_file

    with open(loads_file, "r") as f:
        lines = f.readlines()

    endheader_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            endheader_idx = i
            break

    if endheader_idx is None or endheader_idx + 1 >= len(lines):
        return loads_file

    body_lines = "".join(lines[endheader_idx + 1:])
    df = pd.read_csv(io.StringIO(body_lines), sep=r"\s+")

    if mode == "set_each_hand_kg" and kg_each is not None:
        n_each = float(kg_each) * 9.81
        for side in ["Hand_R_Force", "Hand_L_Force"]:
            y_col = f"{side}_y"
            if y_col in df.columns:
                df[y_col] = -abs(n_each)
    elif mode == "scale_existing" and scale is not None:
        s = float(scale)
        for side in ["Hand_R_Force", "Hand_L_Force"]:
            for axis in ["x", "y", "z"]:
                col = f"{side}_{axis}"
                if col in df.columns:
                    df[col] = df[col] * s

    point_bases = ["Hand_R_pt", "Hand_L_pt", "scapula_R_pt", "scapula_L_pt"]
    for base in point_bases:
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        z_col = f"{base}_z"
        if x_col in df.columns:
            df[x_col] = df[x_col] + shift_x
        if y_col in df.columns:
            df[y_col] = df[y_col] + shift_y
        if z_col in df.columns:
            df[z_col] = df[z_col] + shift_z

    header_lines = lines[: endheader_idx + 1]
    for i, line in enumerate(header_lines):
        low = line.strip().lower()
        if low.startswith("nrows="):
            header_lines[i] = f"nRows={len(df)}\n"
        elif low.startswith("ncolumns="):
            header_lines[i] = f"nColumns={len(df.columns)}\n"

    modified_file = os.path.join(results_dir, "modified_external_force.mot")
    with open(modified_file, "w") as f:
        f.writelines(header_lines)
        df.to_csv(f, sep="\t", index=False, float_format="%.8f", lineterminator="\n")

    return modified_file


def apply_motion_config(motion_file, config, results_dir):
    offsets = config.get("motion_coordinate_offsets_deg", {}) if isinstance(config, dict) else {}
    if not isinstance(offsets, dict) or not offsets:
        return motion_file

    with open(motion_file, "r") as f:
        lines = f.readlines()

    endheader_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            endheader_idx = i
            break

    if endheader_idx is None or endheader_idx + 1 >= len(lines):
        return motion_file

    body_lines = "".join(lines[endheader_idx + 1:])
    df = pd.read_csv(io.StringIO(body_lines), sep=r"\s+")

    changed = False
    for col, delta in offsets.items():
        if col in df.columns:
            df[col] = df[col] + float(delta)
            changed = True

    if not changed:
        return motion_file

    header_lines = lines[: endheader_idx + 1]
    for i, line in enumerate(header_lines):
        low = line.strip().lower()
        if low.startswith("nrows="):
            header_lines[i] = f"nRows={len(df)}\n"
        elif low.startswith("ncolumns="):
            header_lines[i] = f"nColumns={len(df.columns)}\n"

    modified_file = os.path.join(results_dir, "modified_motion.mot")
    with open(modified_file, "w") as f:
        f.writelines(header_lines)
        df.to_csv(f, sep="\t", index=False, float_format="%.8f", lineterminator="\n")

    return modified_file


def collect_additional_outputs(requested_outputs, results_dir, joint_loads, so_force_file, so_activation_file):
    additional_outputs = {}
    requested = requested_outputs if isinstance(requested_outputs, list) else []

    for out_key in requested:
        if out_key in BASE_OUTPUT_KEYS:
            continue

        if out_key == "joint_reaction_all":
            additional_outputs[out_key] = joint_loads
        elif out_key == "all_sto_files":
            all_sto = {}
            for sto_path in glob(os.path.join(results_dir, "*.sto")):
                all_sto[os.path.basename(sto_path)] = read_sto(sto_path)
            additional_outputs[out_key] = all_sto
        elif out_key == "so_force_all":
            additional_outputs[out_key] = read_sto(so_force_file)
        elif out_key == "so_activation_all":
            additional_outputs[out_key] = read_sto(so_activation_file)
        elif isinstance(out_key, str) and out_key.startswith("file:"):
            requested_name = out_key.split("file:", 1)[1].strip()
            file_path = os.path.join(results_dir, requested_name)
            if os.path.exists(file_path):
                additional_outputs[out_key] = read_sto(file_path)
            else:
                additional_outputs[out_key] = f"not found: {requested_name}"

    return additional_outputs


if len(sys.argv) < 4:
    print(json.dumps({"error": "Expected 3 arguments: model_file, motion_file, loads_data_file"}))
    sys.exit(1)

model_file = sys.argv[1]
motion_file = sys.argv[2]
loads_data_file = sys.argv[3]
simulation_config = {}

if len(sys.argv) >= 5:
    try:
        simulation_config = json.loads(sys.argv[4]) if sys.argv[4].strip() else {}
    except Exception as e:
        log(f"Warning: Failed to parse advanced config JSON: {e}")
        simulation_config = {}

results_dir = os.path.abspath(os.path.join('opensim_files', "Output_Folder"))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loads_data_to_use = apply_external_force_config(loads_data_file, simulation_config, results_dir)
motion_file_to_use = apply_motion_config(motion_file, simulation_config, results_dir)

loads_info = [
    ("RightHandForce", "hand_R", "Hand_R_Force", "Hand_R_pt"),
    ("LeftHandForce", "hand_L", "Hand_L_Force", "Hand_L_pt"),
    ("RightScapForce", "scapula_L", "scapula_R_Force", "scapula_R_pt"),
    ("LeftScapForce", "scapula_R", "scapula_L_Force", "scapula_L_pt"),
]

log("Creating External Loads Setup...")
external_loads = osim.ExternalLoads()
external_loads.setDataFileName(loads_data_to_use)

for name, body, force_id, point_id in loads_info:
    force = osim.ExternalForce()
    force.setName(name)
    force.setAppliedToBodyName(body)
    force.setForceIdentifier(force_id)
    force.setPointIdentifier(point_id)
    force.setForceExpressedInBodyName("ground")
    force.setPointExpressedInBodyName("ground")
    external_loads.cloneAndAppend(force)

loads_xml_path = os.path.join(results_dir, "ExternalLoads_Setup.xml")
external_loads.printToXML(loads_xml_path)

log("Loading model and configuring analysis...")
model = osim.Model(model_file)
model.initSystem()

motion = osim.Storage(motion_file_to_use)
initial_time = motion.getFirstTime()
final_time = 0.1

static_optimization = osim.StaticOptimization()
static_optimization.setName("SO_With_Point_Loads")
static_optimization.setStartTime(initial_time)
static_optimization.setEndTime(final_time)
static_optimization.setUseModelForceSet(True)
static_optimization.setUseMusclePhysiology(True)
static_optimization.setActivationExponent(2)
static_optimization.setConvergenceCriterion(0.0001)
static_optimization.setMaxIterations(100)

model.addAnalysis(static_optimization)

log("Setting up Analysis Tool...")
tool = osim.AnalyzeTool(model)
tool.setName("SO_Analysis")
tool.setModel(model)
tool.setInitialTime(initial_time)
tool.setFinalTime(final_time)
tool.setCoordinatesFileName(motion_file_to_use)
tool.setExternalLoadsFileName(loads_xml_path)
tool.setResultsDir(results_dir)
tool.setLoadModelAndInput(True)

log("Running Static Optimization with External Loads...")
tool.run()

so_force_file = os.path.join(results_dir, "SO_Analysis_SO_With_Point_Loads_force.sto")
so_activation_file = os.path.join(results_dir, "SO_Analysis_SO_With_Point_Loads_activation.sto")

log("Running Joint Reaction Analysis...")
model_jr = osim.Model(model_file)
model_jr.initSystem()

jr_analysis = osim.JointReaction()
jr_analysis.setName("JointReactionAnalysis")
jr_analysis.setStartTime(initial_time)
jr_analysis.setEndTime(final_time)
jr_analysis.setForcesFileName(so_force_file)

joint_names = osim.ArrayStr()
on_body_settings = osim.ArrayStr()
in_frame_settings = osim.ArrayStr()

for i in range(model_jr.getJointSet().getSize()):
    joint_names.append(model_jr.getJointSet().get(i).getName())
    on_body_settings.append("child")
    in_frame_settings.append("child")

jr_analysis.setJointNames(joint_names)
jr_analysis.setOnBody(on_body_settings)
jr_analysis.setInFrame(in_frame_settings)

model_jr.addAnalysis(jr_analysis)

jr_tool = osim.AnalyzeTool(model_jr)
jr_tool.setName("JR_Analysis")
jr_tool.setModel(model_jr)
jr_tool.setInitialTime(initial_time)
jr_tool.setFinalTime(final_time)
jr_tool.setCoordinatesFileName(motion_file_to_use)
jr_tool.setExternalLoadsFileName(loads_xml_path)
jr_tool.setResultsDir(results_dir)
jr_tool.setLoadModelAndInput(True)
jr_tool.run()

joint_loads = read_sto(os.path.join(results_dir, "JR_Analysis_JointReactionAnalysis_ReactionLoads.sto"))
spinal_loads = {
    key: value
    for key, value in joint_loads.items()
    if "IVDjnt" in key and any(suffix in key for suffix in ["_fx", "_fy", "_fz"])
} if isinstance(joint_loads, dict) else {}

muscle_forces = read_sto(so_force_file)
muscle_activations = read_sto(so_activation_file)

requested_outputs = simulation_config.get("requested_outputs", BASE_OUTPUT_KEYS)
additional_outputs = collect_additional_outputs(
    requested_outputs,
    results_dir,
    joint_loads,
    so_force_file,
    so_activation_file
)

output_payload = {
    "spinal_loads": spinal_loads,
    "muscle_forces": muscle_forces,
    "muscle_activations": muscle_activations,
    "additional_outputs": additional_outputs,
    "metadata": {
        "requested_outputs": requested_outputs,
        "unapplied_model_modifications": simulation_config.get("model_modifications", [])
    }
}

print(json.dumps(output_payload))

if os.path.exists(results_dir):
    shutil.rmtree(results_dir, ignore_errors=True)
