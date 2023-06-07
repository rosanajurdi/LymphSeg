
from typing import List

import torch
from torch import Tensor, einsum
from ATraining_3D.utils import simplex
import torchio as tio
from Border_Irregularity_Index.Smoothing_Module import GaussianSmoothing





"""
def SmoothBoundary(x, thresh_width=5):
    '''
    Boundary smoothing module
    returns the smoothed groundtruth
    '''

    GS = GaussianSmoothing(1, (3, 3), 100, 2)
    CroporPad = tio.CropOrPad((1, 256, 256))
    smooth = torch.zeros(x.shape)
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        # plt.figure()
        # plt.imshow(smooth[0][0].detach().numpy())
        contour_line = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        # noyeaux = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1))
        # x =plt.imshow(x[0 torch.nn.functional.relu(gs - contour_line)
        x = GS(contour_line)
        x_resized = CroporPad(x.unsqueeze(0))[0]
        x = (x > 0.2) * 1.00
        # x = torch.nn.functional.max_pool2d(x*-1, (2, 2), 1, 1)*-1
        # plt.imshow(x[0][0]*1.00)

    return x
"""
class BorderIrregularityLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = 1
    def __call__(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        pc = net_output
        gt = target
        #pc = net_output[:, self.idc, ...].type(torch.float32)
        #gt = target[:, self.idc, ...].type(torch.float32)
        with torch.no_grad():
            smooth = SmoothBoundary((gt > 0.5) * 1.00) + gt - gt * SmoothBoundary((gt > 0.5) * 1.00)
        # print('pc_dist.shape: ', pc_dist.shape)
        pred_H_index = 2 * (pc * smooth) / (pc + smooth - pc * smooth + 10e-7)
        GT_H_index = 2 * (gt * smooth) / (gt + smooth - gt * smooth + 10e-7)
        borderI_error = (pred_H_index.mean(axis=(2, 3)) - GT_H_index.mean(axis=(2, 3))) ** 2

        hd_loss = borderI_error.sum()

        return hd_loss

class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss