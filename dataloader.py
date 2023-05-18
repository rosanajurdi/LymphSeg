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
from typing import Any, Callable, List, Tuple


spatial_transforms = {
    tio.RandomMotion(): 0.2,
    tio.RandomGhosting(): 0.8,
}
#tio.ToCanonical(),
transforms = tio.Compose([tio.ZNormalization(),tio.EnsureShapeMultiple(2**2),])

# to figure out !!!

transform = tio.Compose(transforms)

def Lymphoma_CREATESPLITS_Dataset(MRI_DATA,GT_DATA, P_TSV):
        patientdir = Path(MRI_DATA)
        labeldir = Path(MRI_DATA, GT_DATA)
        participant_tsv_path = Path(MRI_DATA, P_TSV)
        participants_tsv = pd.read_table(participant_tsv_path).set_index('participant_id')

        label_paths = list(sorted(labeldir.rglob('*.nii.gz')))
        image_paths = list(sorted(set(patientdir.rglob('*.nii.gz')) - set(label_paths)))

        D1 =  participants_tsv[participants_tsv['partition']=='D1']
        D2=  participants_tsv[participants_tsv['partition']=='D2']
        D3 = participants_tsv[participants_tsv['partition']=='D3']
        D4 = participants_tsv[participants_tsv['partition'] == 'D4']

        main_ds = pd.concat([D1, D3])

        train_d1 = D1.sample(frac = 0.7, random_state = 42)
        train_d3 = D3.sample(frac = 0.7, random_state = 42)
        test_d1 = D1.drop(train_d1.index)
        test_d3 = D3.drop(train_d3.index)

        train_ds = pd.concat([train_d1 , train_d3])
        test_ds = pd.concat([test_d1, test_d3])
        artf_ds = pd.concat([D2, D4])

        assert set(test_ds.index).isdisjoint(train_ds.index)
        train_dict = []
        test_dict = []
        #assert len(image_paths) == len(label_paths)
        for train_patient in train_ds.index:
            try:
                LABEL = list(sorted(labeldir.rglob('{}_*'.format(train_patient))))
                MRI = list(sorted(set(patientdir.rglob('{}_*'.format(train_patient))) - set(LABEL)))

                entry = {'MRI': str(MRI[0]), 'LABEL': str(LABEL[0])}

                train_dict.append(entry)
            except:
                print(train_patient)
                pass


        for test_patient in test_ds.index:
            LABEL = list(sorted(labeldir.rglob('{}_*'.format(test_patient))))
            MRI = list(sorted(set(patientdir.rglob('{}_*'.format(test_patient))) - set(LABEL)))

            entry = {'MRI': str(MRI[0]), 'LABEL': str(LABEL[0])}

            test_dict.append(entry)

        dataset_description = {
            "description": 'T1W Lymphoma dataset',
            "labels": {"0": 'background',
                       "1": "Tumor"},

            "modality": "T1w MRI",

            "numTest": len(test_dict),
            "numTrain": len(train_dict),

            "test": test_dict,
            "train": train_dict,
            }

        save_file = open("train_description.json", "w")
        json.dump(dataset_description, save_file, indent=4)
        save_file.close()


class Lymphoma_Dataset:
    def __init__(self, MRI_DATA,GT_DATA, P_TSV, JSON_SPLITFILE):


        participant_tsv_path = Path(MRI_DATA, P_TSV)
        self.participants_tsv = pd.read_table(participant_tsv_path).set_index('participant_id')

        f = open(JSON_SPLITFILE)
        dataset = json.load(f)

        self.train_participants = self.get_participants(dataset['train'])
        self.testing_participants = self.get_participants(dataset['test'])

        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datadatasets()



        print("Data setup: training MRIs : {}, validation MRIs : {}, testing MRIs: {}".format(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)))



    def get_participants(self, IMAGE_LABEL_SET) -> list:
        participants = []
        for Image_LABEL in IMAGE_LABEL_SET:
            patient_path = Image_LABEL["MRI"]
            label_path = Image_LABEL["LABEL"]

            subject = tio.Subject(mri=tio.ScalarImage(patient_path),
                                  gt=tio.LabelMap(label_path),
                                  idd=Path(patient_path).parts[5],
                                  partition=self.participants_tsv['partition'][Path(patient_path).parts[5]], )
            participants.append(subject)

        return participants

    def get_datadatasets(self):
        seed = 3
        torch.manual_seed(seed)
        n = len(self.train_participants)
        r = np.int32(0.8 * n)
        self.traindata, self.valdata = random_split(self.train_participants, [r, n - r])
        train_dataset = tio.SubjectsDataset(self.traindata, transform=transform)
        val_dataset = tio.SubjectsDataset(self.valdata, transform=transform)
        test_dataset = tio.SubjectsDataset(self.testing_participants, transform=transform)


        return train_dataset, val_dataset, test_dataset








