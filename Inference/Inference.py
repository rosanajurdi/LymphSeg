import os, sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append("/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Training")

def get_identifier(filename):
    # Extracts the identifier number from the file name
    return int(os.path.splitext(os.path.basename(filename))[0].split('_')[1])


def Get_3D_patient(patient_path, patient):
    patient_slices = list(sorted(
        patient_path.rglob('{}_*.npy'.format(patient.stem.split('.nii')[0])), key=get_identifier
                                )
                        )

    return patient_slices


def run(args: argparse.Namespace) -> None:
    net_path : str = args.model_path
    set_path : str = args.set_path

    model = torch.load(net_path, map_location=torch.device('cpu'))

    nifty_path = Path(set_path,"input_nifty")
    Prediction = Path(set_path, 'predictions')
    pred_png = Path(set_path, 'pred_png')
    Prediction.mkdir(parents=True, exist_ok=True)
    pred_png.mkdir(parents=True, exist_ok=True)

    patient_list = list(sorted(nifty_path.rglob('*.nii.gz')))
    for patient in patient_list:

        patient_slices = Get_3D_patient(Path(set_path, 'input_npy'), patient)

        patient_predpath = Path(Prediction,patient.stem.split('.nii')[0])
        patient_predpath.mkdir(parents=True, exist_ok=True)

        for slice in tqdm(patient_slices, desc = '{} Progress Bar'.format(str(patient.stem))):
            image = np.load(slice.__str__())
            image = image.reshape(1, 1, image.shape[1], image.shape[2])
            image = torch.tensor(image, dtype=torch.float)
            predicted_slice = model(image)
            predicted_slice = F.softmax(predicted_slice, dim=1).cpu().detach().numpy()
            np.save(Path(patient_predpath,slice.stem), predicted_slice.argmax(axis = 1))




























def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', default = "/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Results/Results_b8/best2-f2.pkl")
    parser.add_argument('--set_path', default = "/Users/rosana.eljurdi/Datasets/LymphSeg_dataset/train")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run(get_args())
