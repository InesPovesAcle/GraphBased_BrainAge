#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:44:22 2026

@author: ines
"""

#!/usr/bin/env python3
import os
import glob
import csv
import numpy as np
import nibabel as nib

# Input directory (uses your $WORK env var)
work = os.environ.get("WORK")
if work is None:
    raise RuntimeError("Environment variable WORK is not set. In bash: export WORK=/path/to/work")

mask_dir = os.path.join(work, "ines", "data", "brain_masks", "ADDecode")
pattern = os.path.join(mask_dir, "*.nii.gz")
mask_paths = sorted(glob.glob(pattern))

if not mask_paths:
    raise RuntimeError(f"No .nii.gz files found at: {pattern}")

out_csv = os.path.join(os.path.join(work, "ines/results/", "icv_from_masks.csv"))

def compute_icv(mask_path: str):
    img = nib.load(mask_path)
    data = img.get_fdata(dtype=np.float32)  # safe for various datatypes
    zooms = img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    # Count voxels inside mask. If mask is probabilistic, >0 works; if strict binary, also fine.
    n_vox = int(np.count_nonzero(data > 0))

    icv_mm3 = n_vox * voxel_vol_mm3
    icv_ml = icv_mm3 / 1000.0
    return n_vox, voxel_vol_mm3, icv_mm3, icv_ml

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "n_voxels_gt0", "voxel_vol_mm3", "icv_mm3", "icv_ml"])

    for p in mask_paths:
        n_vox, voxvol, icv_mm3, icv_ml = compute_icv(p)
        writer.writerow([os.path.basename(p), n_vox, f"{voxvol:.6f}", f"{icv_mm3:.3f}", f"{icv_ml:.3f}"])

print(f"Processed {len(mask_paths)} masks")
print(f"Wrote: {out_csv}")