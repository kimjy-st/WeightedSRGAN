import numpy as np
import cv2
import torch
import torchvision.transforms as T

def get_freq_score(srimg, r):
    graysrimg = cv2.cvtColor(srimg,cv2.COLOR_BGR2GRAY)
    h,w = graysrimg.shape

    dft = cv2.dft(np.float32(graysrimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    row, col = int(h/2), int(w/2)

    HPF = np.ones((h,w,2), np.uint8)
    HPF = cv2.circle(HPF, (row,col),r,0,-1,cv2.LINE_4)
    #HPF[row-rate:row+rate, col-rate:col+rate] = 0  이건 사각형

    HPF_shift = dft_shift * HPF
    HPF_isshift = np.fft.ifftshift(HPF_shift)
    HPF_img = cv2.idft(HPF_isshift)
    HPF_img = cv2.magnitude(HPF_img[:,:,0],HPF_img[:,:,1])

    HPF_out = 20*np.log(cv2.magnitude(HPF_shift[:,:,0],HPF_shift[:,:,1]))

    sum = 0
    num = 0
    for i in range(row):
        for j in range(col):
            if np.abs(HPF_out[i][j]) != float("inf"):
                num+=1
                sum += np.abs(HPF_out[i][j])

    if num == 0:
        mean = "No high freq"
        cv2.imwrite('./ex.png', graysrimg)
    else:
        mean = sum/num
    return mean       


def image_crop(sr, gt, patchsize = 32):
    range_w = (int)(gt.width/patchsize)
    range_h = (int)(gt.height/patchsize)
    freqlist = []
    srtensorlist = []
    for w in range(range_w):
        for h in range(range_h):
            bbox = (w*patchsize, h*patchsize, (w+1)*patchsize, (h+1)*patchsize)
            gt_patch = gt.crop(bbox)
            sr_patch = sr.crop(bbox)
            srtensor = T.ToTensor()(sr_patch)
            srtensorlist.append(srtensor)
            arr_patch = np.array(gt_patch)
            freq = get_freq_score(arr_patch, 4)
            freqlist.append(freq)

    return srtensorlist, freqlist
    



def freq_weight1_ss(freq):
    if freq == "No high freq":
        w = 1e-6
    elif float(freq) < 60:
        w = 1e-6
    elif float(freq) >= 60 and freq < 65:
        w = 2e-6
    elif float(freq) >=65 and freq <100:
        w = 3e-6
    elif float(freq) >=100 and freq < 110:
        w = 4e-6
    elif float(freq) >=110:
        w = 5e-6

    return w


def freq_weight1(freq):
    if freq == "No high freq":
        w = 0.0004
    elif float(freq) < 60:
        w = 0.0004
    elif float(freq) >= 60 and freq < 65:
        w = 0.0006
    elif float(freq) >=65 and freq <100:
        w = 0.0008
    elif float(freq) >=100 and freq < 110:
        w = 0.001
    elif float(freq) >=110:
        w = 0.0015

    return w

def freq_weight1_1(freq):
    if freq == "No high freq":
        w = 0.0006
    elif float(freq) < 60:
        w = 0.0006
    elif float(freq) >= 60 and freq < 65:
        w = 0.0008
    elif float(freq) >=65 and freq <100:
        w = 0.001
    elif float(freq) >=100 and freq < 110:
        w = 0.0012
    elif float(freq) >=110:
        w = 0.0014




