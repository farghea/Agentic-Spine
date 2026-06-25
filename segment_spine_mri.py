
"""
Spine MRI segmentation pipeline.
Usage: python spine_segment.py 4_t2.mha
All outputs → results_medical_image_segmentation/
"""

import os, sys, subprocess, csv
import importlib.util, pathlib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VERTEBRAE_LABELS = {
    1:"sacrum", 2:"L5", 3:"L4", 4:"L3", 5:"L2", 6:"L1",
    7:"T12", 8:"T11", 9:"T10", 10:"T9", 11:"T8", 12:"T7",
    13:"T6", 14:"T5", 15:"T4", 16:"T3", 17:"T2", 18:"T1",
    19:"C7", 20:"C6", 21:"C5", 22:"C4", 23:"C3", 24:"C2", 25:"C1",
}
DISC_LABEL_IN_TOTAL_MR = 20
SPINE_ORDER = [25,24,23,22,21,20,19,
               18,17,16,15,14,13,12,11,10,9,8,7,
               6,5,4,3,2,1]
DISC_BASE   = 30
RESULTS_DIR = "results_medical_image_segmentation"


def convert_mha_to_nii(input_mha, results_dir):
    print("=== Step 1: converting .mha → .nii.gz ===")
    stem    = os.path.splitext(os.path.basename(input_mha))[0]
    out_nii = os.path.join(results_dir, stem + ".nii.gz")
    img     = sitk.ReadImage(input_mha)
    sitk.WriteImage(img, out_nii)
    print(f"  saved: {out_nii}  |  size: {img.GetSize()}  |  spacing: {img.GetSpacing()}")
    return img, out_nii, stem


def run_segmentation(input_nii, stem, results_dir):
    print("\n=== Step 2: running TotalSegmentator ===")
    seg_vert = os.path.join(results_dir, stem + "_vertebrae.nii.gz")
    seg_tot  = os.path.join(results_dir, stem + "_total_mr.nii.gz")
    seg_out  = os.path.join(results_dir, stem + "_seg.nii.gz")

    if os.path.exists(seg_vert):
        print("  [1/2] vertebrae_mr already segmented (cached).")
    else:
        print("  [1/2] vertebrae_mr ...")
        subprocess.run(["TotalSegmentator", "-i", input_nii, "-o", seg_vert,
                        "--ml", "-ta", "vertebrae_mr", "--fast"], check=True)

    if os.path.exists(seg_tot):
        print("  [2/2] total_mr (for discs) already segmented (cached).")
    else:
        print("  [2/2] total_mr (for discs) ...")
        subprocess.run(["TotalSegmentator", "-i", input_nii, "-o", seg_tot,
                        "--ml", "-ta", "total_mr", "--fast"], check=True)

    vert_img = sitk.ReadImage(seg_vert)
    vert_arr = sitk.GetArrayFromImage(vert_img)
    tot_arr  = sitk.GetArrayFromImage(sitk.ReadImage(seg_tot))

    disc_mask   = (tot_arr == DISC_LABEL_IN_TOTAL_MR)
    disc_labels = np.zeros_like(vert_arr)
    disc_map    = {}

    disc_sitk = sitk.GetImageFromArray(disc_mask.astype(np.uint8))
    disc_sitk.CopyInformation(vert_img)
    cc        = sitk.ConnectedComponent(disc_sitk)
    cc_arr    = sitk.GetArrayFromImage(cc)
    n_discs   = int(cc_arr.max())

    for comp_id in range(1, n_discs + 1):
        comp_mask = (cc_arr == comp_id)
        comp_sitk = sitk.GetImageFromArray(comp_mask.astype(np.uint8))
        comp_sitk.CopyInformation(vert_img)
        dilated   = sitk.BinaryDilate(comp_sitk, kernelRadius=[3, 3, 1])
        dil_arr   = sitk.GetArrayFromImage(dilated).astype(bool)

        neighbour_labels = set(vert_arr[dil_arr & (vert_arr > 0)].tolist())
        ordered = [v for v in SPINE_ORDER if v in neighbour_labels]

        if len(ordered) >= 2:
            name = f"{VERTEBRAE_LABELS[ordered[0]]}-{VERTEBRAE_LABELS[ordered[1]]}"
        elif len(ordered) == 1:
            name = f"{VERTEBRAE_LABELS[ordered[0]]}-disc"
        else:
            name = f"disc_{comp_id}"

        label_id = DISC_BASE + comp_id
        disc_labels[comp_mask] = label_id
        disc_map[label_id]     = name

    merged = vert_arr.copy()
    merged[disc_labels > 0] = disc_labels[disc_labels > 0]
    merged_img = sitk.GetImageFromArray(merged)
    merged_img.CopyInformation(vert_img)
    sitk.WriteImage(merged_img, seg_out)

    print(f"  merged segmentation saved: {seg_out}")
    print("  discs found and named:")
    for lid, dname in sorted(disc_map.items()):
        print(f"    [{lid}] {dname}")

    return merged_img, seg_out, disc_map


def compute_disc_centroids(seg_img, disc_map, mid_slice):
    """
    mid_slice is an index into the numpy array's LAST axis (axis 2).
    seg_arr shape is (rows, cols, z) where z = arr.shape[2].
    """
    seg_arr  = sitk.GetArrayFromImage(seg_img)
    n_z      = seg_arr.shape[2]
    mid_safe = min(mid_slice, n_z - 1)   # clamp just in case

    def sort_key(item):
        name = item[1]
        top_name = name.split("-")[0]
        for i, v in enumerate(SPINE_ORDER):
            if VERTEBRAE_LABELS.get(v) == top_name:
                return i
        return 999

    centroids = []
    for label_id, name in sorted(disc_map.items(), key=sort_key):
        mask_3d = (seg_arr == label_id)
        if not mask_3d.any():
            continue

        coords = np.argwhere(mask_3d)       # N×3: (row, col, z)
        c_row  = float(coords[:, 0].mean())
        c_col  = float(coords[:, 1].mean())
        c_z    = float(coords[:, 2].mean())

        mask_2d = mask_3d[:, :, mid_safe]
        if mask_2d.any():
            coords_2d = np.argwhere(mask_2d)
            c2_row = float(coords_2d[:, 0].mean())
            c2_col = float(coords_2d[:, 1].mean())
        else:
            c2_row, c2_col = None, None

        centroids.append({
            "name":   name,
            "c_row":  c_row, "c_col": c_col, "c_z": c_z,
            "c2_row": c2_row, "c2_col": c2_col,
        })

    return centroids


def save_centroids_csv(centroids, mid_slice, stem, results_dir):
    csv_path = os.path.join(results_dir, f"{stem}_disc_centroids_z{mid_slice}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["disc", "row", "col", "z_3d_centroid"])
        writer.writeheader()
        for c in centroids:
            writer.writerow({
                "disc":          c["name"],
                "row":           round(c["c2_row"], 2) if c["c2_row"] is not None else "N/A",
                "col":           round(c["c2_col"], 2) if c["c2_col"] is not None else "N/A",
                "z_3d_centroid": round(c["c_z"],    2),
            })
    print(f"  centroids CSV saved: {csv_path}")
    return csv_path


def plot_middle_slice(img, seg_img, stem, results_dir, disc_map, centroids, mid_slice):
    print("\n=== Step 3: plotting middle slice ===")
    arr     = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    # clamp mid_slice to numpy array bounds
    n_z      = arr.shape[2]
    mid_safe = min(mid_slice, n_z - 1)

    mri_s = arr[:, :, mid_safe].astype(float)
    seg_s = seg_arr[:, :, mid_safe]

    lo, hi   = np.percentile(mri_s, 1), np.percentile(mri_s, 99)
    mri_norm = np.clip((mri_s - lo) / (hi - lo + 1e-8), 0, 1)

    all_labels = {**VERTEBRAE_LABELS, **disc_map}
    n_labels   = max(all_labels.keys()) + 1
    cmap       = plt.cm.get_cmap("tab20b", n_labels)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # panel 1 — raw
    axes[0].imshow(mri_norm, cmap="gray", origin="lower")
    axes[0].set_title(f"Raw MRI  (z={mid_safe})", fontsize=11)
    axes[0].axis("off")

    # panel 2 — segmentation
    axes[1].imshow(seg_s, cmap=cmap, origin="lower", vmin=0, vmax=n_labels)
    axes[1].set_title(f"Segmentation  (z={mid_safe})", fontsize=11)
    axes[1].axis("off")

    # panel 3 — overlay
    axes[2].imshow(mri_norm, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(seg_s == 0, seg_s),
                   cmap=cmap, alpha=0.5, origin="lower", vmin=0, vmax=n_labels)
    axes[2].set_title(f"Overlay  (z={mid_safe})", fontsize=11)
    axes[2].axis("off")

    # panel 4 — disc centroids connected
    axes[3].imshow(mri_norm, cmap="gray", origin="lower")
    visible = [(c["c2_col"], c["c2_row"], c["name"])
               for c in centroids if c["c2_row"] is not None]
    if visible:
        cols = [v[0] for v in visible]
        rows = [v[1] for v in visible]
        axes[3].plot(cols, rows, color="cyan", linewidth=1.5, zorder=2)
        for col, row, name in visible:
            axes[3].scatter(col, row, color="red", s=40, zorder=3)
            axes[3].text(col + 3, row, name, color="yellow",
                         fontsize=6, va="center", zorder=4)
    axes[3].set_title(f"Disc centroids  (z={mid_safe})", fontsize=11)
    axes[3].axis("off")

    present = sorted(set(seg_s.flatten()) - {0})
    patches = [
        mpatches.Patch(color=cmap(lbl / n_labels),
                       label=all_labels.get(lbl, str(lbl)))
        for lbl in present
    ]
    if patches:
        fig.legend(handles=patches, loc="lower center",
                   ncol=min(len(patches), 10), fontsize=7, framealpha=0.8)

    fig.suptitle(f"{stem} — vertebrae + named discs | Raw · Seg · Overlay · Centroids",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    out_png = os.path.join(results_dir, stem + "_middle_slice.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  saved: {out_png}")


def export_middle_slices(img, seg_img, stem, results_dir, mid_slice):
    print("\n=== Step 4: exporting middle slice .nii.gz ===")
    n_z      = sitk.GetArrayFromImage(img).shape[2]
    mid_safe = min(mid_slice, n_z - 1)

    def _extract(sitk_img, tag):
        sz = list(sitk_img.GetSize()); sz[2] = 0
        ex = sitk.ExtractImageFilter()
        ex.SetSize(sz); ex.SetIndex([0, 0, mid_safe])
        fname = os.path.join(results_dir, f"{stem}_{tag}_z{mid_safe}.nii.gz")
        sitk.WriteImage(ex.Execute(sitk_img), fname)
        print(f"  saved: {fname}")

    _extract(img,     "mri")
    _extract(seg_img, "seg")


def _load_matching_module():
    """Dynamically load match_mri_to_model.py regardless of working directory."""
    here = pathlib.Path(__file__).parent
    mod_path = here / "opensim_files" / "match_mri_to_model.py"
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("match_mri_to_model", mod_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pipeline(input_mha):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    img, input_nii, stem = convert_mha_to_nii(input_mha, RESULTS_DIR)
    seg_img, _, disc_map = run_segmentation(input_nii, stem, RESULTS_DIR)

    # mid_slice from numpy array shape (z is last axis = shape[2])
    arr       = sitk.GetArrayFromImage(img)
    mid_slice = arr.shape[2] // 2

    print("\n=== Step 3a: computing disc centroids ===")
    centroids = compute_disc_centroids(seg_img, disc_map, mid_slice)
    save_centroids_csv(centroids, mid_slice, stem, RESULTS_DIR)

    plot_middle_slice(img, seg_img, stem, RESULTS_DIR, disc_map, centroids, mid_slice)
    export_middle_slices(img, seg_img, stem, RESULTS_DIR, mid_slice)

    # ── Step 5: Match MRI to closest OpenSim model ──────────────────────────
    matching_mod = _load_matching_module()
    if matching_mod is not None:
        try:
            matches = matching_mod.run_matching(centroids, k=5)
            if matches:
                # Save results next to the centroid CSV
                out_csv = os.path.join(
                    RESULTS_DIR, f"{stem}_model_matches_z{mid_slice}.csv"
                )
                import csv as _csv
                with open(out_csv, "w", newline="") as f:
                    fieldnames = [
                        "rank", "distance_deg", "n_shared_levels",
                        "orientation", "sex", "age_group",
                        "lumbar_lordosis_deg", "thoracic_kyphosis_deg",
                        "model_file", "shared_levels",
                    ]
                    writer = _csv.DictWriter(
                        f, fieldnames=fieldnames, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for m in matches:
                        m["shared_levels"] = "|".join(m["shared_levels"])
                        writer.writerow(m)
                print(f"  Model match results saved: {out_csv}")
        except Exception as e:
            print(f"  WARNING: Model matching failed — {e}")
    else:
        print("  WARNING: match_mri_to_model.py not found; skipping model matching.")

    print(f"\n=== All done! Everything saved in ./{RESULTS_DIR}/ ===")


if __name__ == "__main__":
    run_pipeline('4_t2.mha')
