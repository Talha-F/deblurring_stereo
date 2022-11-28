import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
from skimage.metrics import structural_similarity as ssim

def SSIM(output, target):
    output = output.clamp(0.0,1.0)
    return ssim(output, target, data_range=1.0, multichannel=True)
	
def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)

def RMSE(output, target):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    return math.sqrt(mse)
