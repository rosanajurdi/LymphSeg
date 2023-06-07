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
from Smoothing_Module import GaussianSmoothing, BINARIZED_GaussianSmoothing

def gt2Smooth(segm: Tensor):
    segm = torch.tensor(segm).reshape(1, 1, im.shape[0], im.shape[1])
    bi_p = bi_n = 0
    p = 0
    bi_index = BI_Index()
    for x in range(10):
        bi_p = bi_n
        GS = BINARIZED_GaussianSmoothing(1, 10, 2 ** x, dim=2)
        sm = GS(segm)
        bi_n = bi_index(sm, segm)
        if bi_p == bi_n:
            break
    return 2**(x - 1), sm
#
def pred2smooth(pred: Tensor, sigma: int):
    pred = torch.tensor(pred).reshape(1, 1, im.shape[0], im.shape[1])
    GS = GaussianSmoothing(1, 10, sigma, dim=2)
    sm = GS(pred)

    return sm

#        p = p + 1

class BI_Index():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = 1
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, smooth: Tensor, target: Tensor) -> Tensor:
        pc = smooth.type(torch.float32)
        tc = target.type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc) # a.a*
        union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))  # a* + a
        divided: Tensor = (union - 2 * einsum("bc->b", intersection)) / (einsum("bcwh->bc", pc) )
        #divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) ) / (einsum("bc->b", union) )

        loss = divided.mean()

        return loss


class BI_Index_2():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = 1
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, smooth: Tensor, target: Tensor) -> Tensor:
        pc = smooth.type(torch.float32)
        tc = target.type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc) # a.a*
        union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))  # a* + a
        #divided: Tensor = (union - 2 * einsum("bc->b", intersection)) / (einsum("bcwh->bc", pc) )
        divided: Tensor = 1 - 2*(einsum("bc->b", intersection) ) / (einsum("bc->b", union) )

        loss = divided.mean()

        return loss

def BI_Loss(pred: Tensor, segm: Tensor):
    ...


segmentation_maps = list(sorted(root.rglob('sub-1075883_*.npy')))

segmentation_map = Path(root, 'sub-1075883_100.npy')

im = np.load(segmentation_map)

plt.imsave( "/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Border_Irregularity_Index/images/im.pdf", im, cmap=cm.gray)
plt.imshow(im, cmap=cm.gray)
plt.show()

BI_1 = BI_Index()
BI_2 = BI_Index_2()



sigma , s_gt  = gt2Smooth(torch.tensor(im))

s_pred = pred2smooth(torch.tensor(im), sigma)
I_b = BI_2(torch.tensor(im).reshape(1, 1, 184, 184), s_gt)





""" 
for s in [1, 2, 8, 16, 32, 64, 128, 200, 250]:
    GF = GaussianSmoothing(1, 10 ,s, dim = 2 )
    s_image = GF(torch.tensor(im).reshape(1, 1, im.shape[0], im.shape[1]))
    plt.imshow(s_image.detach().numpy()[0][0], cmap=cm.gray)
    plt.show()
    assert len(np.unique(s_image)) == 2 and  len(np.unique(im)) == 2
    assert im.shape == s_image[0][0].shape
    BId = BI_1(torch.tensor(im).reshape(1, 1, im.shape[0], im.shape[1]), s_image)
    BId_2 = BI_2(torch.tensor(im).reshape(1, 1, im.shape[0], im.shape[1]), s_image)
    print(s, BId.numpy()*100, BId_2*100)
    BI_Loss()
    plt.imsave("/Users/rosana.eljurdi/PycharmProjects/LymphSeg1/Border_Irregularity_Index/images/im_{}_{}.pdf".format(s, BId),
               s_image.detach().numpy()[0][0],
               cmap=cm.gray
               )
    plt.show()

print(segmentation_maps)

"""