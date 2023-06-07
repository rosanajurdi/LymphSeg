import sys
sys.path.append("/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/ATraining_3D")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import shutil
import nibabel as nib
from Border_Irregularity_Index import Smoothing_Module

import torch
from typing import List
from torch import Tensor, einsum
from ATraining_3D.utils import simplex
import torchio as tio
root = Path("/Users/rosana.eljurdi/Datasets/LymphSeg_dataset/test/gt_npy")

class BI_Index():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = 1
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs.type(torch.float32)
        tc = target.type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) ) / (einsum("bc->b", union) )

        loss = divided.mean()

        return loss




segmentation_maps = list(sorted(root.rglob('sub-1075883_*.npy')))



BID = BI_Index()

for segm in segmentation_maps:
    segmentation_map = Path(segm)

    im = np.load(segmentation_map)

    plt.imsave("/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Border_Irregularity_Index/images/im.pdf", im,
               cmap=cm.gray)
    plt.imshow(im, cmap=cm.gray)
    plt.show()

    for s in [1, 2, 8, 16, 32, 64, 128, 200, 250]:
        GF = Smoothing_Module.GaussianSmoothing(1, 10 ,s, dim = 2 )

        s_image = GF(torch.tensor(im).reshape(1, 1, im.shape[0], im.shape[1]))

        plt.imshow(s_image.detach().numpy()[0][0], cmap=cm.gray)
        plt.show()


        assert len(np.unique(s_image)) == 2 and  len(np.unique(im)) == 2
        assert im.shape == s_image[0][0].shape

        BI = np.round((im.sum() - s_image.sum() )*100/s_image.sum(), 2)

        BI_diff = BID(torch.tensor(im).reshape(1, 1, im.shape[0], im.shape[1]), s_image)

        print(s, BI.numpy(), BI_diff.numpy()*100)

        #plt.imsave("/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Border_Irregularity_Index/images/im_{}_{}_{}.pdf".format(s, BI, BI_diff),
        #           s_image.detach().numpy()[0][0],
        #           cmap=cm.gray
        #           )
        plt.show()

print(segmentation_maps)