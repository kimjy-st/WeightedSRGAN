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
only_test_y_channel = True
# Model architecture name
g_arch_name = "srresnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
num_rcb = 16
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "srresnet_x4_+pretrained"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"/home/jykim/Project/improved/data/DIV2K_GT"

    test_gt_images_dir = f"/home/jykim/Project/improved/data/Set14/GTmod12"
    test_lr_images_dir = f"/home/jykim/Project/improved/data/Set14/LRbicx4"

    gt_image_size = 96
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = f"/home/jykim/Project/improved/SRResNet/weight/SRResNet_x4-ImageNet-6dd5216c.pth.tar"

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs (1,000,000 iters)
    epochs = 90

    # loss function weights
    loss_weights = 1.00

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address

    set5dir = ["/home/jykim/Project/improved/data/Set5/LRbicx4","/home/jykim/Project/improved/SRResNet/SR_images_resnet/Set5","/home/jykim/Project/improved/data/Set5/GTmod12"]
    set14dir = ["/home/jykim/Project/improved/data/Set14/LRbicx4", "/home/jykim/Project/improved/SRResNet/SR_images_resnet/Set14","/home/jykim/Project/improved/data/Set14/GTmod12"]
    BSD100dir = ["/home/jykim/Project/improved/data/BSDS100/LRbicx4","/home/jykim/Project/improved/SRResNet/SR_images_resnet/BSD100","/home/jykim/Project/improved/data/BSDS100/GTmod12"]
    manga109dir = ["/home/jykim/Project/improved/data/Manga109/LRbicx4","/home/jykim/Project/improved/SRResNet/SR_images_resnet/Manga109","/home/jykim/Project/improved/data/Manga109/GTmod12"]
    urban100dir = ["/home/jykim/Project/improved/data/Urban100/LRbicx4","/home/jykim/Project/improved/SRResNet/SR_images_resnet/Urban100","/home/jykim/Project/improved/data/Urban100/GTmod12"]

    lr_dir =urban100dir[0]
    sr_dir =urban100dir[1]
    gt_dir =urban100dir[2]

    model_weights_path = "/home/jykim/Project/results/srresnet_x4_+pretrained/g_best.pth.tar"