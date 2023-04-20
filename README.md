# WeightedSRGAN

# Baseline and Dataset

https://github.com/Lornatang/SRGAN-PyTorch

- srgan과 srresnet train, test, config 모두 있음

### Train Dataset

- Imagenet (Pretrained) + DIV2K

### Test Dataset

- Set5
- Set14
- BSD100
- Manga109
- Urban100

# Baseline training and test

## 1. SRGAN baseline 성능 획득

1. srgan_config_ori.py 에서 mode = test 변경
2. g_model_weights_path =/weights/SRGAN_x4-ImageNet-8c4a7569.pth.tar 로 설정 (위 깃허브에 pretrained weight 다운받기)

```python
g_model_weights_path = "/weights/SRGAN_x4-ImageNet-8c4a7569.pth.tar
```

1. Super Resolution된 image 저장할 파일 설정
2. test dataset 바꿔가며 SR_image 얻기
3. metrics 파일에서 save_metrics.py 실행 

## 2. test시 성능 측정

```python
pip install pyiqa
pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
```

[https://github.com/S-aiueo32/lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch)

- 설치 후 save_metrics.py 에서 sr_image 저장된 path 와 결과 파일을 저장할 savedirrroot 변경
- 데이터셋 별로 이미지의 psnr, ssim, lpips 성능이 기록된 csv 파일이 저장됨

## 3. SRResNet baseline 성능 획득

1. srresnet_config.py 에서 mode = train 설정
2. pretrained_model_weights_path = /weights/SRResNet_x4-ImageNet-6dd5216c.pth.tar 로 설정
3. train_gt_images_dir = DIV2K_GT 로 설정
4. train_srresnet.py 실행
    

---

1. srresnet_config.py에서 mode = test  설정
2. model_weights_path = “./results/{exp_name}/g_best.pth.tar” 설정
3. sr_image가 저장될 경로 설정 
4. lr_dir, sr_dir, gt_dir 바꿔가며 성능 구하기 
5. 만들어진 sr_images 가지고 위의 save_metrics.py 실행

