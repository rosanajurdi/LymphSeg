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


brain_mask = "/Users/rosana.eljurdi/Datasets/defacing-templates/tpl-MNI152Lin_res-01_desc-brain_mask.nii.gz"
input = "/Users/rosana.eljurdi/Datasets/defacing-templates/tpl-MNI152Lin_res-01_T1w.nii.gz"


transforms = tio.Compose([tio.ToCanonical(),
                          #tio.Mask(masking_method = 'mask',outside_value = 0),
                        tio.EnsureShapeMultiple(2**2),
                    ])

# to figure out !!!

transform = tio.Compose(transforms)

def Lymphoma_CREATESPLITS_Dataset(MRI_DATA,GT_DATA, P_TSV, ROOT):
    "function responsible for splitting the dataset and generating a json file with the splits"

    patientdir = Path(MRI_DATA)
    labeldir = Path(MRI_DATA, GT_DATA)
    participant_tsv_path = Path(MRI_DATA, P_TSV)
    participants_tsv = pd.read_csv(participant_tsv_path, delimiter = ',').set_index('participant_id')

    label_paths = list(sorted(labeldir.rglob('*.nii.gz')))
    image_paths = list(sorted(set(patientdir.rglob('*.nii.gz')) - set(label_paths)))

    D1 =  participants_tsv[participants_tsv['partition']=='D1']
    D2=  participants_tsv[participants_tsv['partition']=='D2']
    D3 = participants_tsv[participants_tsv['partition']=='D3']
    D4 = participants_tsv[participants_tsv['partition'] == 'D4']

    train_d1 = D1.sample(frac = 0.7, random_state = 42)
    train_d3 = D3.sample(frac = 0.7, random_state = 42)

    test_d1 = D1.drop(train_d1.index)
    test_d3 = D3.drop(train_d3.index)

    val_d1 = train_d1.sample(frac = 0.2, random_state = 42)
    val_d3 = train_d3.sample(frac=0.2, random_state=42)

    train_d1 = train_d1.drop(val_d1.index)
    train_d3 = train_d3.drop(val_d3.index)

    train_ds = pd.concat([train_d1 , train_d3])
    val_ds = pd.concat([val_d1 , val_d3])
    test_ds = pd.concat([test_d1, test_d3])
    artf_ds = pd.concat([D2, D4])

    assert set(test_ds.index).isdisjoint(train_ds.index) and set(val_ds.index).isdisjoint(train_ds.index)
    train_dict = []
    test_dict = []
    val_dict = []
    art_dict = []
    #assert len(image_paths) == len(label_paths)

    for train_patient in train_ds.index:
        try:
            LABEL = list(sorted(labeldir.rglob('{}_*'.format(train_patient))))
            MRI = list(sorted(set(patientdir.rglob('{}_*'.format(train_patient))) - set(LABEL)))
            entry = {'MRI': str(MRI[0].relative_to(ROOT)),
                    'LABEL': str(LABEL[0].relative_to(ROOT))}

            train_dict.append(entry)
        except:
            print(train_patient)
            pass


    for test_patient in test_ds.index:
        LABEL = list(sorted(labeldir.rglob('{}_*'.format(test_patient))))
        MRI = list(sorted(set(patientdir.rglob('{}_*'.format(test_patient))) - set(LABEL)))

        entry = {'MRI': str(MRI[0].relative_to(ROOT)),
                 'LABEL': str(LABEL[0].relative_to(ROOT))}

        test_dict.append(entry)


    for validation_patient in val_ds.index:
        LABEL = list(sorted(labeldir.rglob('{}_*'.format(validation_patient))))
        MRI = list(sorted(set(patientdir.rglob('{}_*'.format(validation_patient))) - set(LABEL)))

        entry = {'MRI': str(MRI[0].relative_to(ROOT)),
                 'LABEL': str(LABEL[0].relative_to(ROOT))}

        val_dict.append(entry)


    for art_patient in artf_ds.index:
        LABEL = list(sorted(labeldir.rglob('{}_*'.format(art_patient))))
        MRI = list(sorted(set(patientdir.rglob('{}_*'.format(art_patient))) - set(LABEL)))

        entry = {'MRI': str(MRI[0].relative_to(ROOT)),
                 'LABEL': str(LABEL[0].relative_to(ROOT))}

        art_dict.append(entry)



    dataset_description = {
        "description": 'T1W Lymphoma dataset',
        "labels": {"0": 'background',
                   "1": "Tumor"},

        "modality": "T1w MRI",

        "numTest": len(test_dict),
        "numTrain": len(train_dict),
        "numval": len(val_dict),
        "numArtf": len(art_dict),

        "test": test_dict,
        "train": train_dict,
        "val": val_dict,
        "artf": art_dict
        }

    save_file = open("train_description.json", "w")
    json.dump(dataset_description, save_file, indent=4)
    save_file.close()


class Lymphoma_Dataset:
    def __init__(self, MRI_DATA,GT_DATA, P_TSV, JSON_SPLITFILE):

        participant_tsv_path = Path(MRI_DATA, P_TSV)
        self.participants_tsv = pd.read_table(participant_tsv_path, delimiter= ',').set_index('participant_id')

        self.MRI_DATA = MRI_DATA
        self.GT_DATA = GT_DATA

        f = open(JSON_SPLITFILE)
        dataset = json.load(f)

        self.train_participants = self.get_participants(dataset['train'])
        self.testing_participants = self.get_participants(dataset['test'])
        self.validation_participants = self.get_participants(dataset['val'])
        self.artf_participants = self.get_participants(dataset['artf'])

        self.train_dataset = tio.SubjectsDataset(self.train_participants, transform=transform)
        self.val_dataset = tio.SubjectsDataset(self.validation_participants, transform=transform)
        self.test_dataset = tio.SubjectsDataset(self.testing_participants, transform=transform)
        self.artf_dataset = tio.SubjectsDataset(self.artf_participants, transform=transform)



        print("Data setup: training MRIs : {}, validation MRIs : {}, testing MRIs: {}".format(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)))



    def get_participants(self, IMAGE_LABEL_SET) -> list:
        participants = []
        for Image_LABEL in IMAGE_LABEL_SET:
            patient_path = Image_LABEL["MRI"]
            label_path = Image_LABEL["LABEL"]

            subject = tio.Subject(mri=tio.ScalarImage(Path(self.MRI_DATA,patient_path)),
                                  gt=tio.LabelMap(Path(self.MRI_DATA, label_path)),
                                  idd=Path(patient_path),
                                  partition=self.participants_tsv['partition'][Path(patient_path).stem.split('_ses')[0]],
                                  mask = tio.LabelMap(brain_mask),
                                  )
            participants.append(subject)

        return participants


















