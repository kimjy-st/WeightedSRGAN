import os
import pyiqa
import numpy as np
import torchvision.transforms as T
from PIL import Image
import cv2
import math
import torch
from torch import Tensor
from lpips_pytorch import LPIPS, lpips
from IQA_pytorch import SSIM, utils

######################################################
################ PSNR ###############################
#####################################################

def PSNR_score(hrimage, srimage):
        """ hrimage : PIL image
            srimage : PIL image """
        srimage = np.array(srimage)
        hrimage = np.array(hrimage)
        if hrimage.shape != srimage.shape:
            print("size not equal")
        
        else:
            mse = np.mean((hrimage-srimage)**2)
            if mse ==0:
                return ("Can't calculate / mse = 0")
            else:
                return 20* math.log10(255.0 / math.sqrt(mse))



#########################################################3
###################### SSIM ###############################
##########################################################

def ssim(hrimage, srimage):
    C1= (0.01 *255)**2
    C2 = (0.03 * 255)**2
 
    hrimage = hrimage.astype(np.float64)
    srimage = srimage.astype(np.float64)

    kernel = cv2.getGaussianKernel(11,1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(hrimage, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(srimage, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(hrimage**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(srimage**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(hrimage * srimage, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()




def calculate_ssim(hrimage, srimage):
    srimage = np.array(srimage)
    hrimage = np.array(hrimage)

    if not hrimage.shape == srimage.shape:
        print("'Input images must have the same dimensions")
    if hrimage.ndim == 2:
        return ssim(hrimage, srimage)
    elif hrimage.ndim == 3:
        if hrimage.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(hrimage,srimage))
            return np.array(ssims).mean()
        elif hrimage.shape[2] == 1:
            return ssim(np.squeeze(hrimage), np.squeeze(srimage))

#######################################################################
####################### LPIPS #########################################
#######################################################################



def LPIPSscore(hrimage, srimage):
    srtensor = T.ToTensor()(srimage)
    hrtensor = T.ToTensor()(hrimage)

    criterion = LPIPS( net_type='vgg',  version='0.1')

    loss = criterion(srtensor, hrtensor)
    loss = lpips(srtensor, hrtensor, net_type='vgg', version='0.1')

    return loss.item()


#######################################################################
########################## NIQE #######################################
#######################################################################

def NIQE(srroot, device = "cuda:3"):
    srimage = Image.open(srroot)
    srtensor = T.ToTensor()(srimage)
    srtensor = srtensor.unsqueeze(0)

    niqe = pyiqa.create_metric('niqe', device = device)
    niqe_score = niqe(srtensor, color_space = 'ycbcr')

    return niqe_score.item()

##################################################################
##################### FID #######################################
#################################################################

def FID(srdirroot, hrdirroot, device = "cuda:3"):
    fid = pyiqa.create_metric('fid', device = device)
    fid_score = fid(srdirroot, hrdirroot)
    
    return(fid_score.item())


#################################################################
##################### SSIM Library ##############################
#################################################################

def SSIM_by_lib(hrimage, srimage):

    srimage = utils.prepare_image(srimage)
    hrimage = utils.prepare_image(hrimage)

    model = SSIM(channels = 3)

    ssim_score = model(srimage, hrimage, as_loss = False)

    return ssim_score.item()

def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """Check if the dimensions of the two tensors are the same
    Args:
        raw_tensor (np.ndarray or torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
    """
    # Check if tensor scales are consistent
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"
def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch
    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`
    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel
    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format
    """
    if only_use_y_channel:
        weight = Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[65.481, -37.797, 112.0],
                         [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 0), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor

def _psnr_torch(raw_arr, dst_arr, 
                only_test_y_channel: bool, crop_border = 4) -> float:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function
    Args:
        raw_path (gt path): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_path (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics
    """
    # Check if two tensor scales are similar

    raw_tensor = T.ToTensor()(raw_arr).unsqueeze(0)
    dst_tensor = T.ToTensor()(dst_arr).unsqueeze(0)

   

    _check_tensor_shape(raw_tensor, dst_tensor)

    # crop border pixels
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Convert RGB tensor data to YCbCr tensor, and extract only Y channel data
    if only_test_y_channel:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    
    mse_value = torch.mean((raw_tensor * 255.0 - dst_tensor * 255.0) ** 2 + 1e-8, dim=[1, 2, 3])
    psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)

    return round(psnr_metrics.item(),4)


