# Deblurring Stereo

This repository is for deblurring model. Our goal is to find more high-performing methods to remove blurring from stereo video and create clear images in real-time. 

<p align="center">
  <img width=100% src="https://user-images.githubusercontent.com/33317140/207792447-77962acf-1ed1-455b-a1f1-2a811e86c052.png">
</p>


## Stereo Blur Dataset
<p align="center">
  <img width=95% src="https://user-images.githubusercontent.com/33317140/207792444-b49c6a00-253a-4857-a11d-e8ab5424edb5.png">
</p>
We built a dataset from the [[KITTI Stereo Dataset]](https://www.cvlibs.net/datasets/kitti/eval_stereo.php) applying blurring method from [[DeblurGAN]](https://github.com/KupynOrest/DeblurGAN).

This data has patterns of real blurred images from the driving scene. 

You can download the dataset (2.06G, unzipped 2.06G) from [[Dropbox]](https://www.dropbox.com/s/12h9b4u5eysi5k0/kitti_final_788_Split.tar).

You have to unzip files and make following structure of directories.

├── input \
├──── Test \
├────── disparity_left \
├────── disparity_right \
├────── image_left \
├────── image_right \
├────── image_left_blur_ga \
├────── image_right_blur_ga \
├──── Train \
├────── disparity_left \
├────── disparity_right \
├────── image_left \
├────── image_right \
├────── image_left_blur_ga \
└────── image_right_blur_ga \

You can make disparity maps with `disparity_gen.ipynb`.

## Pretrained Models

You could download the pretrained model (34.8MB) of DAVANet which is our baseline from [[Here]](https://drive.google.com/file/d/1oVhKnPe_zrRa_JQUinW52ycJ2EGoAcHG/view?usp=sharing). 

(Note that the model does not need to unzip, just load it directly.)

## Prerequisites

- Linux (tested on Ubuntu 14.04/16.04)
- Python 2.7+
- Pytorch 0.4.1

#### Installation

```
conda env create -f environment.yml
```

## Get Started

Use the following command to train the neural network:

```
python runner.py 
        --phase 'train'\
        --data [dataset path]\
        --out [output path]
```

Use the following command to test the neural network:

```
python runner.py \
        --phase 'test'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
Use the following command to resume training the neural network:

```
python runner.py 
        --phase 'resume'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
You can also use the following simple command, with changing the settings in config.py:

```
python runner.py
```

You can see the training log from tensorboard.

```
tensorboard --logdir=[logdir path]
```
<!-- 
## Results on the testing dataset

<p align="center">
  <img width=100% src="https://user-images.githubusercontent.com/14334509/57179916-ea446e80-6eb5-11e9-8eb6-98fb878810e7.png">
</p>
-->
## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to seohyeol@andrew.cmu.edu

## Reference
[1] Zhou, Shangchen, et al. Davanet: Stereo deblurring with view aggregation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[2] Zhe Hu, Li Xu, and Ming-Hsuan Yang. Joint depth estimation and camera shake removal from single blurry image. In *CVPR*, 2014.

[3] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In *CVPR*, 2017.

[4] Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, and Jiri Matas. Deblurgan: Blind motion deblurring using conditional adversarial networks. In CVPR, 2018.

[5] Jiawei Zhang, Jinshan Pan, Jimmy Ren, Yibing Song, Lin- chao Bao, Rynson WH Lau, and Ming-Hsuan Yang. Dynamic scene deblurring using spatially variant recurrent neural networks. In *CVPR*, 2018. 

[6] Xin Tao, Hongyun Gao, Xiaoyong Shen, Jue Wang, and Jiaya Jia. Scale-recurrent network for deep image deblurring. In *CVPR*, 2018.

[7] Kim, Kiyeon, Seungyong Lee, and Sunghyun Cho. MSSNet: Multi-Scale-Stage Network for Single Image Deblurring. arXiv preprint arXiv:2202.09652, 2022.

[8] Kupyn, Orest, et al. Deblurgan: Blind motion deblurring using conditional adversarial networks. Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

## License

This project is open sourced under MIT license.
