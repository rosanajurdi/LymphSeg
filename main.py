# This is a sample Python script.
import argparse
import torch

import torchio as tio
import torch
import numpy as np
import torchio as tio

from torch.utils.data import DataLoader,random_split
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torchio as tio
import pandas as pd
import json
from tqdm import tqdm



class Lymphoma_Erroned():
    def __init__(self, Dataset_path):
        dictt = []
        self.Dataset_path = Dataset_path
        self.LABEL_DATA = list(sorted(Path(Dataset_path).rglob('*_ses-M000_dseg.nii.gz')))
        self.MRI_DATA = sorted(set(list(sorted(Path(Dataset_path).rglob('*.nii.gz')))) - set(self.LABEL_DATA))
        for patients in zip(self.MRI_DATA,self.LABEL_DATA) :
            entry = {'MRI': str(patients[0]),
                    'LABEL': str(patients[1])}

            dictt.append(entry)

        transforms_Tocanon = tio.Compose([tio.ToCanonical(),])
        self.participants = self.get_participants(dictt)
        self.dataset = tio.SubjectsDataset(self.participants, transform=transforms_Tocanon)

        target_folder = "/Users/rosana.eljurdi/Datasets/err/Lymphoma_oriented2"
        for patient in tqdm(self.dataset):
            patient['mri'].save(Path(target_folder, patient['idd']))
            patient['gt'].save(Path(target_folder, patient['idgt']))


    def get_participants(self, IMAGE_LABEL_SET) -> list:
        participants = []
        for Image_LABEL in IMAGE_LABEL_SET:
            patient_path = Image_LABEL["MRI"]
            label_path = Image_LABEL["LABEL"]

            subject = tio.Subject(mri=tio.ScalarImage(Path(patient_path)),
                                  gt=tio.LabelMap(Path(label_path)),
                                  idd=Path(Path(patient_path).relative_to(self.Dataset_path)),
                                  idgt=Path(Path(label_path).relative_to(self.Dataset_path)),
                                  )
            participants.append(subject)

        return participants









def run(args: argparse.Namespace) -> None:
    MRI_DATA: str = args.dataset
    GT_DATA: str = args.manual_seg

    ROOT = args.data_root_folder

    lymphdataset = Lymphoma_Erroned(ROOT).dataset



    """ 
    for batch in dataset:

        inputs = batch['mri']['data'].type(torch.float)
        labels = batch['gt']['data'].type(torch.float)

        print(inputs.max(), inputs.min(), input().mean())
    """




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument('--dataset', type=str,
                        default='/Users/rosana.eljurdi/Datasets/err/Lymphoma_one_patient')
    parser.add_argument('--manual_seg', type=str,
                        default='derivatives/manual_segm')

    parser.add_argument("--workdir", type=str,
                        default='where-to-save-results')

    parser.add_argument('--data_root_folder', type=str,
                        default="/Users/rosana.eljurdi/Datasets/err/Lymphoma_one_patient")

    parser.add_argument('--create_splits', type = bool, default = False)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    return args





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 3
    torch.manual_seed(seed)

    run(get_args())


