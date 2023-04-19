import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda:3")
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "srresnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
num_rcb = 16
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "SRGAN_x4-DIV2K_freqw1_1"

if mode == "test":
    # Test data address #freqw1_1
    #0 : lr, 1: sr, 2: gt
    set5dir = ["/home/jykim/Project/improved/data/Set5/LRbicx4","/home/jykim/Project/improved/0418/SR_images/Set5","/home/jykim/Project/improved/data/Set5/GTmod12"]
    set14dir = ["/home/jykim/Project/improved/data/Set14/LRbicx4", "/home/jykim/Project/improved/0418/SR_images/Set14","/home/jykim/Project/improved/data/Set14/GTmod12"]
    BSD100dir = ["/home/jykim/Project/improved/data/BSDS100/LRbicx4","/home/jykim/Project/improved/0418/SR_images/BSD100","/home/jykim/Project/improved/data/BSDS100/GTmod12"]
    manga109dir = ["/home/jykim/Project/improved/data/Manga109/LRbicx4","/home/jykim/Project/improved/0418/SR_images/Manga109","/home/jykim/Project/improved/data/Manga109/GTmod12"]
    urban100dir = ["/home/jykim/Project/improved/data/Urban100/LRbicx4","/home/jykim/Project/improved/0418/SR_images/Urban100","/home/jykim/Project/improved/data/Urban100/GTmod12"]

    test_dir = [set5dir, set14dir,BSD100dir, manga109dir, urban100dir]
    g_model_weights_path = f"/home/jykim/Project/improved/SRGAN/results/SRGAN_x4-DIV2K_freqw1_1/g_best.pth.tar"

    lr_dir = set14dir[0]

    sr_dir = set14dir[1]
    gt_dir = set14dir[2]


    