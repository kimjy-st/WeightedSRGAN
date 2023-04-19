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
