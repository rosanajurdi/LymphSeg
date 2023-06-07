# This is a sample Python script.
import argparse
from functools import partial
from typing import Any, Callable, List, Tuple
from pathlib import Path
import torch
from ATraining_3D import utils
import nibabel as nb
import torch.nn as nn
from dataloader import Lymphoma_Dataset, Lymphoma_CREATESPLITS_Dataset
from torch.utils.data import DataLoader
from ATraining_3D import networks
import ATraining_3D.losses as ls
from torch import Tensor
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ATraining_3D import metrics
from monai.networks.nets import UNet
from monai.losses.dice import DiceLoss


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = norm

    return res


def run(args: argparse.Namespace) -> None:
    MRI_DATA: str = args.dataset
    GT_DATA: str = args.manual_seg
    P_TSV: str = args.participant_tsv
    JSON_FILE: str = args.split_file
    create_splits: str = args.create_splits
    ROOT = args.data_root_folder

    # Loading the Bids format dataset
    lymphdataset = Lymphoma_Dataset(MRI_DATA, GT_DATA, P_TSV, JSON_FILE)
    lymphdatatrain = lymphdataset.train_dataset
    lymphdatatest = lymphdataset.test_dataset
    lymphdataval = lymphdataset.val_dataset
    lymphdataarttest = lymphdataset.artf_dataset

    # Creating the new data
    LymphSeg = Path("/Users/rosana.eljurdi/Datasets/LymphSeg")
    Train = Path(LymphSeg, 'train')
    Test = Path(LymphSeg, 'test')
    Val = Path(LymphSeg, "val")
    Artf = Path(LymphSeg, 'artf_test')
    # Create Directories

    for set_file in [Train, Val,Test, Artf]:
        # Making directory
        LymphSeg.mkdir(parents=True, exist_ok=True)
        set_file.mkdir(parents=True, exist_ok=True)
        Path(set_file, 'input_npy').mkdir(parents=True, exist_ok=True)
        Path(set_file, 'gt_npy').mkdir(parents=True, exist_ok=True)
        Path(set_file, 'input_nifty').mkdir(parents=True, exist_ok=True)
        Path(set_file, 'gt_nifty').mkdir(parents=True, exist_ok=True)


    # Getting Loaders
    import matplotlib.pyplot as plt
    import os
    for lymphdata, compartment in enumerate(zip([lymphdatatrain, lymphdataval, lymphdatatest, lymphdataarttest],
                                                [Train, Val, Test, Artf ])):
        lymphdata = compartment[0]
        folder = compartment[1]

        for batch in tqdm(lymphdata, desc = '{} Progress Bar'.format(str(folder.stem))):
            input_nifty = torch.transpose(batch['mri'].data, 1, 2)[0]
            gt_nifty = torch.transpose(batch['gt'].data, 1, 2)[0]
            path_to_save = Path(folder, 'input_npy').__str__()
            gt_path_to_save = Path(Path(folder, 'gt_npy')).__str__()

            batch['mri'].save(Path(folder, 'input_nifty', "{}.nii.gz".format(batch['idd'].stem.split("_ses")[0])).__str__())
            batch['gt'].save(Path(folder, 'gt_nifty', "{}.nii.gz".format(batch['idd'].stem.split("_ses")[0])).__str__())

            for i, slice in enumerate(zip(input_nifty, gt_nifty)):

                if len(np.unique(slice[0])) > 1:
                    npy_to_save = norm_arr(slice[0].numpy())
                    assert 0 <= npy_to_save[0].min() and npy_to_save[0].max() <= 1, (npy_to_save[0].min(), npy_to_save[0].max())
                    print(np.unique(slice[1].numpy()))
                    np.save( os.path.join(path_to_save, "{}_{}.npy".format(batch['idd'].stem.split('_ses')[0], i)), npy_to_save.reshape(1, 184, 184))
                    np.save(os.path.join(gt_path_to_save, "{}_{}.npy".format(batch['idd'].stem.split('_ses')[0], i)), slice[1].numpy())





import Get_Args

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 3
    torch.manual_seed(seed)

    run(Get_Args.get_args())


