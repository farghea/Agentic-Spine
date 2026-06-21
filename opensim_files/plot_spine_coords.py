import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import scipy for CubicSpline, otherwise use a polynomial fallback
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 1. Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "initial_coords_all_models.csv")
output_plot = os.path.join(BASE_DIR, "spine_coordinates.png")
csv_norm_path = os.path.join(BASE_DIR, "initial_coords_normalized.csv")
csv_metrics_path = os.path.join(BASE_DIR, "spine_curvature_metrics.csv")

# 2. Read the CSV file
if not os.path.exists(csv_path):
    print(f"CSV file not found at: {csv_path}")
    print("Please run extract_initial_coords.py first to generate the CSV.")
    exit(1)

print(f"Reading data from: {csv_path}")
df = pd.read_csv(csv_path)

# 3. Define the anatomical ordering of the vertebrae (T1 -> T12 -> L1 -> L5)
# This maps to the column names in the CSV file
vertebrae = [f"thoracic{i}" for i in range(1, 13)] + [f"lumbar{i}" for i in range(1, 6)]
labels = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]

# Verify which columns actually exist in the CSV
valid_vertebrae = []
valid_labels = []
for vert, label in zip(vertebrae, labels):
    if f"{vert}_x" in df.columns and f"{vert}_y" in df.columns:
        valid_vertebrae.append(vert)
        valid_labels.append(label)

if not valid_vertebrae:
    print("Error: No vertebral columns found in the CSV file.")
    exit(1)

# Verify required vertebrae are present for Lordosis & Kyphosis
required_metrics = ["lumbar5", "lumbar1", "thoracic12", "thoracic1"]
has_metrics_verts = all(v in valid_vertebrae for v in required_metrics)
if not has_metrics_verts:
    print(f"Warning: Some vertebrae needed for LL/TK ({required_metrics}) are missing.")

# Create the figure with 2x2 subplots for visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
ax1 = axs[0, 0]  # Plot 1: As Is
ax2 = axs[0, 1]  # Plot 2: Normalized & Rotated
ax3 = axs[1, 0]  # Plot 3: Lordosis Histogram
ax4 = axs[1, 1]  # Plot 4: Kyphosis Histogram

# Data structures to store lines and calculated values
raw_lines = {"Female": [], "Male": []}
norm_lines = {"Female": [], "Male": []}
spine_angles = []

# Lists to collect data for the output Excel files
norm_rows = []
metrics_rows = []

female_plotted = False
male_plotted = False

# Loop over each model to process, fit splines, and calculate parameters
for idx, row in df.iterrows():
    x_raw = np.array([row[f"{vert}_x"] for vert in valid_vertebrae], dtype=float)
    y_raw = np.array([row[f"{vert}_y"] for vert in valid_vertebrae], dtype=float)
    
    # Identify sex for grouping
    sex = str(row.get("sex", "")).strip().capitalize()
    if sex not in ["Female", "Male"]:
        sex = "Male"  # Default fallback
        
    color = "#e75480" if sex == "Female" else "#3498db"
    
    # Fit spline to the raw coords (x as a function of y)
    sort_idx = np.argsort(y_raw)
    y_sorted = y_raw[sort_idx]
    x_sorted = x_raw[sort_idx]
    sorted_verts = [valid_vertebrae[i] for i in sort_idx]
    
    if HAS_SCIPY:
        spline_fit = CubicSpline(y_sorted, x_sorted)
        spline_der = spline_fit.derivative()
    else:
        # Fallback to degree 3 polynomial
        poly_coeff = np.polyfit(y_sorted, x_sorted, 3)
        spline_fit = np.poly1d(poly_coeff)
        spline_der = np.polyder(spline_fit)

    # ─── PLOT 1: As Is ───
    label_raw = f"{sex} Models" if (sex == "Female" and not female_plotted) or (sex == "Male" and not male_plotted) else ""
    ax1.plot(x_raw, y_raw, marker='o', markersize=2.5, color=color, alpha=0.15, linewidth=0.8, label=label_raw)
    raw_lines[sex].append((x_raw, y_raw))

    # ─── PLOT 2: Normalized & Rotated ───
    l5_idx = valid_vertebrae.index("lumbar5") if "lumbar5" in valid_vertebrae else -1
    t1_idx = valid_vertebrae.index("thoracic1") if "thoracic1" in valid_vertebrae else 0
    
    x_l5, y_l5 = x_raw[l5_idx], y_raw[l5_idx]
    x_trans = x_raw - x_l5
    y_trans = y_raw - y_l5
    
    x_t1_trans = x_trans[t1_idx]
    y_t1_trans = y_trans[t1_idx]
    
    L = math.sqrt(x_t1_trans**2 + y_t1_trans**2)
    if L < 1e-6:
        L = 1.0
        
    cos_theta = y_t1_trans / L
    sin_theta = x_t1_trans / L
    
    # Rotate & scale (XY coordinates)
    x_norm = (x_trans * cos_theta - y_trans * sin_theta) / L
    y_norm = (x_trans * sin_theta + y_trans * cos_theta) / L
    # Scale Z coordinate uniformly by L
    z_norm = np.array([row[f"{vert}_z"] for vert in valid_vertebrae], dtype=float) / L
    
    label_norm = f"{sex} Models" if (sex == "Female" and not female_plotted) or (sex == "Male" and not male_plotted) else ""
    ax2.plot(x_norm, y_norm, marker='o', markersize=2.5, color=color, alpha=0.15, linewidth=0.8, label=label_norm)
    norm_lines[sex].append((x_norm, y_norm))
    
    # Build row for the normalized coordinates file
    norm_row = {
        "sex": row.get("sex"),
        "age_group": row.get("age_group"),
        "model_file": row.get("model_file")
    }
    for vert, x_n, y_n, z_n in zip(valid_vertebrae, x_norm, y_norm, z_norm):
        norm_row[f"{vert}_x_norm"] = round(x_n, 6)
        norm_row[f"{vert}_y_norm"] = round(y_n, 6)
        norm_row[f"{vert}_z_norm"] = round(z_n, 6)
    norm_rows.append(norm_row)
    
    # ─── LORDOSIS & KYPHOSIS ANGLE CALCULATION ───
    if has_metrics_verts:
        # Get y coordinates of L5, L1, T12, T1
        y_l5_val = row["lumbar5_y"]
        y_l1_val = row["lumbar1_y"]
        y_t12_val = row["thoracic12_y"]
        y_t1_val = row["thoracic1_y"]
        
        # Slopes (dx/dy) at these positions
        slope_l5 = spline_der(y_l5_val)
        slope_l1 = spline_der(y_l1_val)
        slope_t12 = spline_der(y_t12_val)
        slope_t1 = spline_der(y_t1_val)
        
        # Normal angles in degrees (angle of normal relative to horizontal)
        theta_l5 = math.degrees(math.atan(slope_l5))
        theta_l1 = math.degrees(math.atan(slope_l1))
        theta_t12 = math.degrees(math.atan(slope_t12))
        theta_t1 = math.degrees(math.atan(slope_t1))
        
        # Lumbar Lordosis (LL) angle = change in tilt between L5 and L1
        lordosis = abs(theta_l5 - theta_l1)
        # Thoracic Kyphosis (TK) angle = change in tilt between T12 and T1
        kyphosis = abs(theta_t12 - theta_t1)
        
        spine_angles.append({
            "sex": sex,
            "lordosis": lordosis,
            "kyphosis": kyphosis
        })
        
        # Build row for the curvature metrics file (metadata + macro curves)
        metrics_row = {
            "sex": row.get("sex"),
            "age_group": row.get("age_group"),
            "model_file": row.get("model_file"),
            "lumbar_lordosis_deg": round(lordosis, 4),
            "thoracic_kyphosis_deg": round(kyphosis, 4)
        }
        
        # ─── DISCRETE CURVATURE (ANGLES BETWEEN SUCCESSIVE SEGMENTS) ───
        # For each intermediate vertebra (index 1 to n-2), compute the angle between the segment
        # leading into it (i-1 to i) and the segment leading out of it (i to i+1).
        # We sort bottom-to-top (L5 -> L4 -> L3 ... -> T1) to keep logical ordering.
        n_verts = len(sorted_verts)
        for i in range(1, n_verts - 1):
            vert_name = sorted_verts[i]
            
            # Segment vector before: P_{i-1} -> P_i
            u_x = x_sorted[i] - x_sorted[i-1]
            u_y = y_sorted[i] - y_sorted[i-1]
            
            # Segment vector after: P_i -> P_{i+1}
            w_x = x_sorted[i+1] - x_sorted[i]
            w_y = y_sorted[i+1] - y_sorted[i]
            
            # Compute signed angle (degrees) between vector u and vector w
            # Positive = counter-clockwise change, Negative = clockwise change
            dot_p = u_x * w_x + u_y * w_y
            cross_p = u_x * w_y - u_y * w_x
            discrete_angle = math.degrees(math.atan2(cross_p, dot_p))
            
            # Store in row
            metrics_row[f"discrete_angle_{vert_name}_deg"] = round(discrete_angle, 4)
            
        metrics_rows.append(metrics_row)

    if sex == "Female":
        female_plotted = True
    elif sex == "Male":
        male_plotted = True

# Convert angle results to DataFrame
df_angles = pd.DataFrame(spine_angles)

# ─── Save Results to CSV ───
df_norm_out = pd.DataFrame(norm_rows)
df_metrics_out = pd.DataFrame(metrics_rows)

df_norm_out.to_csv(csv_norm_path, index=False)
df_metrics_out.to_csv(csv_metrics_path, index=False)
print(f"Saved normalized coordinates to: {csv_norm_path}")
print(f"Saved curvature metrics to: {csv_metrics_path}")

# ─── Plot Average Curves (Smooth Splines) ───
def plot_average_spline(ax, lines, color, label):
    if not lines:
        return
    mean_x = np.mean([l[0] for l in lines], axis=0)
    mean_y = np.mean([l[1] for l in lines], axis=0)
    
    sort_idx = np.argsort(mean_y)
    my_sorted = mean_y[sort_idx]
    mx_sorted = mean_x[sort_idx]
    
    if HAS_SCIPY:
        mean_spline = CubicSpline(my_sorted, mx_sorted)
    else:
        mean_spline = np.poly1d(np.polyfit(my_sorted, mx_sorted, 3))
        
    y_smooth = np.linspace(my_sorted.min(), my_sorted.max(), 100)
    x_smooth = mean_spline(y_smooth)
    
    ax.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label)
    ax.scatter(mean_x, mean_y, color=color, s=25, zorder=5)

# Add averages to Plot 1 and 2
for sex, color_mean, label_prefix in [("Female", "#c0392b", "Female Mean"), ("Male", "#2980b9", "Male Mean")]:
    plot_average_spline(ax1, raw_lines[sex], color_mean, label_prefix)
    plot_average_spline(ax2, norm_lines[sex], color_mean, label_prefix)

# Add vertebra annotations to Plot 1
mean_source = raw_lines["Male"] if raw_lines["Male"] else raw_lines["Female"]
if mean_source:
    mean_x = np.mean([l[0] for l in mean_source], axis=0)
    mean_y = np.mean([l[1] for l in mean_source], axis=0)
    for lbl, mx, my in zip(valid_labels, mean_x, mean_y):
        ax1.annotate(lbl, (mx, my), textcoords="offset points", xytext=(12, -3), ha='left', fontsize=8, alpha=0.8, fontweight='bold')

# Add vertebra annotations to Plot 2
mean_norm_source = norm_lines["Male"] if norm_lines["Male"] else norm_lines["Female"]
if mean_norm_source:
    mean_x = np.mean([l[0] for l in mean_norm_source], axis=0)
    mean_y = np.mean([l[1] for l in mean_norm_source], axis=0)
    for lbl, mx, my in zip(valid_labels, mean_x, mean_y):
        ax2.annotate(lbl, (mx, my), textcoords="offset points", xytext=(12, -3), ha='left', fontsize=8, alpha=0.8, fontweight='bold')

# ─── PLOT 3 & 4: Histograms for Lordosis and Kyphosis ───
if not df_angles.empty:
    for sex, color in [("Female", "#e75480"), ("Male", "#3498db")]:
        subset = df_angles[df_angles["sex"] == sex]
        if subset.empty:
            continue
            
        # Lordosis Histogram
        ax3.hist(
            subset["lordosis"], 
            bins=12, 
            color=color, 
            alpha=0.6, 
            label=f"{sex} (Mean: {subset['lordosis'].mean():.1f}°, Std: {subset['lordosis'].std():.1f}°)",
            edgecolor='none'
        )
        
        # Kyphosis Histogram
        ax4.hist(
            subset["kyphosis"], 
            bins=12, 
            color=color, 
            alpha=0.6, 
            label=f"{sex} (Mean: {subset['kyphosis'].mean():.1f}°, Std: {subset['kyphosis'].std():.1f}°)",
            edgecolor='none'
        )

# ─── Style Subplots ───
ax1.set_title("1. Spine Coordinates (As Is)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("X Coordinate (Anterior-Posterior) [m]", fontsize=9)
ax1.set_ylabel("Y Coordinate (Vertical) [m]", fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.axis("equal")
ax1.legend(loc="upper right", fontsize=8)

ax2.set_title("2. Normalized & Rotated\n(L5 at (0,0), T1 on Y-axis at (0,1))", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Normalized X [dimensionless]", fontsize=9)
ax2.set_ylabel("Normalized Y [dimensionless]", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.axis("equal")
ax2.legend(loc="upper right", fontsize=8)

ax3.set_title("3. Lumbar Lordosis Distribution (L5 to L1)", fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel("Lordosis Angle [degrees]", fontsize=9)
ax3.set_ylabel("Count / Models", fontsize=9)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="upper right", fontsize=8)

ax4.set_title("4. Thoracic Kyphosis Distribution (T12 to T1)", fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel("Kyphosis Angle [degrees]", fontsize=9)
ax4.set_ylabel("Count / Models", fontsize=9)
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(loc="upper right", fontsize=8)

method_str = "Cubic Spline" if HAS_SCIPY else "Polynomial (Degree 3)"
plt.suptitle(f"Spine Shape & Curvature Metrics (Fitting method: {method_str})", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Save the plot
plt.savefig(output_plot, dpi=300)
print(f"Spine analysis plot saved successfully to: {output_plot}")

# Attempt to show the plot
try:
    plt.show()
except Exception:
    pass
