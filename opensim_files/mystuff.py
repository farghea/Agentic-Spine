


#%% 

import subprocess
import json
import os

# 1. Setup paths


model_path = 'Male/Age4049/985_SizeScaled_CurvatureAdjust_MuscleAdjust.osim'
motion_path = 'NMB_Motion15.mot'
external_force_path = 'NMB_ExternalForce15.mot'



model_file  = os.path.abspath(
    model_path
)
motion_file = os.path.abspath(
    motion_path
)
forces_file = os.path.abspath(
    external_force_path
)


# 2. Build the command as a SINGLE STRING
# We wrap paths in quotes "{variable}" to handle spaces safely
# cmd = f'conda run -n opensim --no-capture-output python worker.py "{model_file}" "{motion_file}" "{forces_file}"'

cmd = f'conda run -n opensim --no-capture-output python opensim_run.py "{model_file}" "{motion_file}" "{forces_file}"'

print("Running OpenSim via conda...")
print(f"Command: {cmd}")

try:
    # 3. Execute and capture output
    # When cmd is a string, we usually need shell=True
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # Check if the command actually worked (return code 0)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

    # 4. Parse JSON
    output_str = result.stdout
    json_start_index = output_str.find('{')
    
    if json_start_index != -1:
        json_data = output_str[json_start_index:]
        data = json.loads(json_data)

        # 5. Get your variables
        spinal_loads = data.get("spinal_loads")
        muscle_forces = data.get("muscle_forces")
        muscle_activations = data.get("muscle_activations")

        print("Success!")
        # print(f"Spinal Loads keys: {list(spinal_loads.keys()) if spinal_loads else 'None'}")
    else:
        print("Error: No JSON data found in output.")
        print("Raw Output:\n", output_str)

except subprocess.CalledProcessError as e:
    print("Error: The OpenSim process failed.")
    print("Error Log:", e.stderr)
except json.JSONDecodeError as e:
    print("Error: Failed to decode the JSON response.")
    print("Raw Output:", result.stdout)


