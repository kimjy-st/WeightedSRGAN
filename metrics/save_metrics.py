from metrics import _psnr_torch,LPIPSscore,SSIM_by_lib
import os
import csv
from PIL import Image
def save_metrics(hrdirroot, srdirroot, savedirroot):
    results = []
    for file in os.listdir(srdirroot):
        imagename = file

        srimageroot = os.path.join(srdirroot, imagename)
        hrimageroot = os.path.join(hrdirroot, imagename)

        hrimageroot = Image.open(hrimageroot)
        srimageroot = Image.open(srimageroot)

        psnr_score = _psnr_torch(hrimageroot, srimageroot, only_test_y_channel=True)
        ssim_score = SSIM_by_lib(srimageroot, hrimageroot)

        LPIPS_score = LPIPSscore(srimageroot, hrimageroot)
    
        result = [imagename, psnr_score ,ssim_score,LPIPS_score]
        results.append(result)
        l = len(os.listdir(srdirroot))
        print("Get metrics!  remain : ",l-len(results))

    listname = ["imagename", "PSNR", "SSIM", "LPIPS"]
    with open(savedirroot,"a",newline = '') as f:
        wr = csv.writer(f)
        wr.writerow(listname) 

    for i in range(len(results)): 
        with open(savedirroot,"a",newline = '') as f:
            wr = csv.writer(f)
            wr.writerow(results[i])
#"../SRGAN/SR_images/Set5","../SRGAN/SR_images/Set14","../SRGAN/SR_images/BSD100","../freqw2/SR_images/Manga109","../freqw2/SR_images/Urban100"]
SRGAN_sr = ["../srresnet_baseline/SR_images/Set5","../srresnet_baseline/SR_images/Set14","../srresnet_baseline/SR_images/BSD100","../srresnet_baseline/SR_images/Manga109","../srresnet_baseline/SR_images/Urban100"]

GT_dirs = ["/home/jykim/Project/improved/data/Set5/GTmod12","/home/jykim/Project/improved/data/Set14/GTmod12","/home/jykim/Project/improved/data/BSDS100/GTmod12",
           "/home/jykim/Project/improved/data/Manga109/GTmod12","/home/jykim/Project/improved/data/Urban100/GTmod12"]
#"/home/jykim/Project/improved/data/Set5/GTmod12","/home/jykim/Project/improved/data/Set14/GTmod12","/home/jykim/Project/improved/data/BSDS100/GTmod12",

for i in range(len(SRGAN_sr)):
    gtdirroot = GT_dirs[i] 
    srdirroot = SRGAN_sr[i]
    datasetname = SRGAN_sr[i].split("/")[-1]
    savedirroot = f"../srresnet_baseline/srresnet_baseline_{datasetname}result.csv"

    save_metrics(gtdirroot, srdirroot, savedirroot)














