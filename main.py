# This is a sample Python script.
import argparse
from functools import partial
from typing import Any, Callable, List, Tuple
from pathlib import Path
import torch
from Training import utils
import nibabel as nb
import torch.nn as nn
from dataloader import Lymphoma_Dataset, Lymphoma_CREATESPLITS_Dataset
from torch.utils.data import DataLoader
from Training import networks
import Training.losses as ls
from torch import Tensor
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from Training import metrics
import os
from models.modified3DUnet import Modified3DUNet
def setup(args) :
    print(">>> Setting up")
    # configuring GPU settings:
    cpu: bool = not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")
    # Data loaders:
    # network
    model = Modified3DUNet(1, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    bi_criterion = ls.BorderIrregularityLoss()
    loss_fns = criterion
    print(model, optimizer, loss_fns, device)
    return model, optimizer, loss_fns, device

def do_epoch(mode: str, model: Any, device: Any, loader: DataLoader, epc: int,
             loss_fnc:Any, optimizer : Any = None) -> Tuple[float, float]:

    assert mode in ['train', 'val']
    log_loss_per_epoch, metrics_per_epoch = 0, []
    if mode == 'train':
        model.train()
        desc = f">> Training   ({epc})"
    elif mode == 'val':
        model.eval()
        desc = f">> Validation ({epc})"

    total_iteration , total_images = len(loader), len(loader.dataset)
    Dice_metric = 0
    for j, batch in tqdm(enumerate(loader)):
        inputs = batch['mri']['data'].type(torch.float)
        labels = batch['gt']['data'].type(torch.float)

        if optimizer:
            optimizer.zero_grad()

        # forward pass
        #print(inputs.shape)
        #passs = False
        
        # Checking the space:
        #print(inputs.shape)
        prediction = model(inputs.to(device))
        
        pred_probs: Tensor = F.sigmoid(prediction)
        precition_map : Tensor = pred_probs.round()

        #print(batch['mri']['stem'], inputs.shape, prediction.shape)


        #print(inputs.shape,labels.shape, prediction.shape)
        loss = loss_fnc(labels.squeeze().to(device), prediction.squeeze())
        #print(loss)

        if optimizer:
            loss.backward
            optimizer.step()

        log_loss_per_epoch +=  loss.detach()
        predictions_bool = utils.convert_to_bool(precition_map.detach().cpu().numpy())
        groundtruth_bool = utils.convert_to_bool(labels.detach().cpu().numpy())

        # the dice is added over all the data and the mean is obtained !

        Dice_metric += np.sum(metrics.dice_coef_3d(groundtruth_bool, predictions_bool))

    metrics_per_epoch = Dice_metric/total_images
    print(f'Epoch {epc}, Loss: {loss:.4f}, Dice : {Dice_metric:.4f}')
    return log_loss_per_epoch, metrics_per_epoch, model

def run(args: argparse.Namespace) -> None:
    MRI_DATA : str=args.dataset
    GT_DATA: str=args.manual_seg
    batchsize : int=args.batch_size
    P_TSV : str=args.participant_tsv
    shuffle: bool=args.shuffle
    JSON_FILE : str = args.split_file
    create_splits: str = args.create_splits 


    model, optimizer, loss_func, device = setup(args)
    if create_splits == True:
        Lymphoma_CREATESPLITS_Dataset(MRI_DATA, GT_DATA, P_TSV)
    lymphdataset = Lymphoma_Dataset(MRI_DATA, GT_DATA, P_TSV, JSON_FILE)
    lymphdatatrain = lymphdataset.train_dataset
    lymphdatatest = lymphdataset.test_dataset

    # Getting Loaders

    training_loader = DataLoader(lymphdatatrain, batch_size=args.batch_size, num_workers=1)
    testing_loader = DataLoader(lymphdatatest, batch_size = 1, num_workers=1)

    print("training on {} MRIs".format(len(training_loader.dataset)))

    # initializing weights
    num_epochs = 10

    trall_dice : Tensor = torch.zeros(num_epochs, dtype=torch.float32, device = 'cpu')
    trall_losslog : Tensor = torch.zeros(num_epochs, dtype=torch.float32, device = 'cpu')
    tstall_dice: Tensor = torch.zeros(num_epochs, dtype=torch.float32, device='cpu')
    tstall_losslog: Tensor = torch.zeros(num_epochs, dtype=torch.float32, device='cpu')

    # Train the model
    for epoch in range(num_epochs):
        trall_losslog[epoch], trall_dice[epoch], model = do_epoch(mode='train',model= model,
                 device = device,loader= training_loader,
                epc = epoch,loss_fnc = loss_func, optimizer = optimizer)

        with torch.no_grad():
            tstall_losslog[epoch], tstall_dice[epoch], _ = do_epoch(mode='val', model=model,
                                                           device=device, loader=testing_loader,
                                                           epc=epoch, loss_fnc=loss_func, optimizer=None)

        if tstall_losslog[-1] < tstall_losslog[-2]:
            torch.save(model, Path(args.result_file, 'best-loss.pkl') )
        if tstall_dice[-1] > tstall_dice[-2]:
            torch.save(model, Path(args.result_file, 'best-dice.pkl'))


    training_metadata = \
        {
        "train-loss": trall_losslog, "test-loss": tstall_losslog,
        "train-dice" : trall_dice, "test-Dice" : tstall_dice
        }

    df_metric = pd.DataFrame.from_dict(data=training_metadata)

    Path(args.result_file).mkdir(parents=True, exist_ok=True)
    df_metric.to_csv(Path(args.result_file, 'results_per_epoch.csv'))

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument('--dataset', type=str,
                        default='/network/lustre/iss02/aramis/users/rosana.eljurdi/Lymphoma-defaced-brain-mni')
    parser.add_argument('--manual_seg', type=str,
                        default='derivatives/manual_segm')
    parser.add_argument('--participant_tsv', type=str,
                        default='participants_.tsv')
    parser.add_argument("--result_file", type=str, default='/network/lustre/iss02/aramis/users/rosana.eljurdi/LymphSeg/Results')
    parser.add_argument("--workdir", type=str,
                        default='where-to-save-results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=str,
                        default=True)

    parser.add_argument('--split_file', type=str,
                        default="/network/lustre/iss02/aramis/users/rosana.eljurdi/LymphSeg/train_description.json")


    parser.add_argument('--create_splits', type = bool, default = False)
    args = parser.parse_args()
    print(args)
    return args

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    seed = 3
    torch.manual_seed(seed)
    
    run(get_args())


