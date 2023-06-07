import torch
from surface_distance import metrics
from torch import Tensor
from functools import partial
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor) -> bool:
    return uniq(a).issubset([0,1])


def compute_3D_Dice(GT: Tensor, PR: Tensor) -> Tensor:
    Dice = []
    for i in range(GT.shape[0]):
        Dice.append(metrics.compute_dice_coefficient(mask_gt=GT[i, :, :, :, :],
                                                     mask_pred=PR[i, :, :, :, :]))

    return Dice


dice_coef = partial(compute_3D_Dice, "bcwh->bc")
dice_coef_3d = compute_3D_Dice
# partial(compute_3D_Dice)
