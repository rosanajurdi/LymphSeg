import os
import numpy as np
import nibabel as nb
import pandas as pd
from pathlib import Path
from surface_distance import metrics
from pathlib import Path

modified_paths = []

for path in basenames:
    parts = path.parts
    if len(parts) >= 8:
        new_parts = parts[:6] + ("gt",) + parts[8:]
        modified_path = Path(*new_parts)
        modified_paths.append(modified_path)
    else:
        modified_paths.append(path)

# Print the modified paths
for path in modified_paths:
    print(path)

def get_identifier(filename):
    # Extracts the identifier number from the file name
    return int(os.path.splitext(os.path.basename(filename))[0].split('_')[1])

root = "/Users/rosana.eljurdi/Datasets/LymphSeg_dataset/test"
p_dir = Path(root, "predictions")
nifty_path = Path(root, "input_nifty")
nifty_pred = Path(root, 'pred_nifty')
nifty_pred.mkdir(parents=True, exist_ok=True)

nifty_gt_packed = Path(root, 'packedgt_nifty')
nifty_gt_packed.mkdir(parents=True, exist_ok=True)

patients  = list(sorted(nifty_path.rglob('sub-*')))

# read the nifty file to get the header and the affine:
for patient in patients:

    nft = nb.load(patient)
    print(nft.shape, nft.affine, nft.header)

    p_path = Path(p_dir, patient.stem.split('.nii')[0])
    basenames = list(sorted(p_path.rglob('*.npy'), key=get_identifier))
    Volume_3D_Pred = np.array([np.load(p)[0] for p in basenames])

    gt_basenames = [Path(*(b.parts[:6] + ("gt_npy",) + parts[8:])) for b in basenames]
    Volume_3D_GT = np.array([np.load(p) for p in gt_basenames])

    sf = metrics.compute_surface_distances(np.array(Volume_3D_Pred).astype(bool),
                                           np.array(Volume_3D_GT).astype(bool), [1,1,1])

    Dice_3D = metrics.compute_dice_coefficient(Volume_3D_Pred, Volume_3D_GT)
    Hauss_3d = metrics.compute_robust_hausdorff(sf, 95)
    MASD_3D = metrics.compute_average_surface_distance(sf)
    #Saving the predictions and gt:
    a = nb.Nifti1Image(np.swapaxes(Volume_3D_Pred, 0,1), affine = nft.affine, header = nft.header)
    nb.save(a, Path(nifty_pred, patient.name))

    b = nb.Nifti1Image(np.swapaxes(Volume_3D_GT, 0,1), affine = nft.affine, header = nft.header)
    nb.save(b, Path(nifty_gt_packed, patient.name))











p_dir = Path("/Users/rosana.eljurdi/Datasets/LymphSeg_dataset/val/predictions")


gt_paths = list(sorted(gt_dir.rglob('sub-*')))
p_paths = gt_paths.str









nifty_gt = "/Users/rosana.eljurdi/Datasets/LymphSeg_dataset/val/gt_nifty/sub-1070629.nii.gz"


