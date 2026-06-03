import requests
import json
import traceback
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
from mhr_pose_info import pose_info
import base64


def sam3d_call(image_path):
    result_dir = "sam 3d body result"
    os.makedirs(result_dir, exist_ok=True)

    # Helper function to download a file from a URL
    def download_file(url, folder=result_dir):
        if not url: return
        try:
            filename = url.split("/")[-1]
            filepath = os.path.join(folder, filename)
            print(f"Downloading {filename}...")
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"Failed to download {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    with open("info_and_keys.json", "r") as f:
        keys = json.load(f)

    # Read and base64-encode the local image
    # image_path = "lift_test.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Could not find image '{image_path}'")
        exit(1)

    with open(image_path, "rb") as image_file:
        # Segmind API usually requires the base64 string directly
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    url = "https://api.segmind.com/v1/sam-3d-body"
    headers = {
        "x-api-key": keys.get("segmind_api_key", ""),
        "Content-Type": "application/json"
    }

    data = {
        "image": encoded_string,
        "include_keypoints": True,
        "return_individual_meshes": True
    }

    try:
        print("Sending request to Segmind... (this might take a few seconds)")
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            print("Success! Saving results to JSON...")
            json_path = os.path.join(result_dir, "sam3d_results.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
                
            print("Downloading meshes and keypoints...")
            # Download URLs found in the result
            download_file(result.get("model_glb"))
            download_file(result.get("keypoints"))
            download_file(result.get("keypoints_3d"))
            
            for kp_url in result.get("individual_keypoints", []):
                download_file(kp_url)
                
            metadata = result.get("metadata", {})
            for mesh in metadata.get("meshes", []):
                download_file(mesh.get("mesh_ply"))
                
            print("All downloads complete! Check the 'sam 3d body result' folder.")

        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
    
    return None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def plot_keypoints_3d_matplotlib(json_path):
    if not os.path.exists(json_path):
        print(f"Cannot find {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # the json has format {"num_people": 1, "people": [{"person_id": 0, "keypoints_3d": [[x,y,z], ...]}]}
    keypoints_list = []
    if isinstance(data, dict):
        if "people" in data:
            for p in data["people"]:
                if "keypoints_3d" in p:
                    keypoints_list.append(p["keypoints_3d"])
        elif "keypoints_3d" in data:
            keypoints_list = data["keypoints_3d"]
    elif isinstance(data, list):
        keypoints_list = data
        
    if not keypoints_list:
        print("No keypoints found in JSON.")
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    skeleton = pose_info.get("skeleton_info", {})
    keypoint_info = pose_info.get("keypoint_info", {})
    name_to_idx = {v['name']: v['id'] for k, v in keypoint_info.items() if 'name' in v and 'id' in v}
    
    for pts_array in keypoints_list:
        pts = np.array(pts_array)
        if len(pts.shape) != 2 or pts.shape[1] != 3:
            continue
            
        # Coordinates: x, y, z
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        
        ax.scatter(x, z, -y, c='blue', marker='o', s=10)
        
        for k, v in skeleton.items():
            n1, n2 = v['link']
            if n1 in name_to_idx and n2 in name_to_idx:
                idx1 = name_to_idx[n1]
                idx2 = name_to_idx[n2]
                if idx1 < len(pts) and idx2 < len(pts):
                    p1 = pts[idx1]
                    p2 = pts[idx2]
                    color = np.array(v.get('color', [0, 0, 0])) / 255.0
                    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('-Y')
    plt.title("3D Keypoints")
    
    # Make axis exactly equal
    x_lims = ax.get_xlim3d()
    y_lims = ax.get_ylim3d()
    z_lims = ax.get_zlim3d()
    
    x_range = abs(x_lims[1] - x_lims[0])
    x_middle = np.mean(x_lims)
    y_range = abs(y_lims[1] - y_lims[0])
    y_middle = np.mean(y_lims)
    z_range = abs(z_lims[1] - z_lims[0])
    z_middle = np.mean(z_lims)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    # Also enforce box aspect if supported
    try:
        ax.set_box_aspect([1.0, 1.0, 1.0])
    except Exception:
        pass

    plt.show()

if __name__ == "__main__":
    # sam3d_call("lift_test.jpg") 
    # plot_keypoints_3d_matplotlib("sam 3d body result/keypoints_3d.json")
    pass