# SQAD

This is the implementation of the paper [SQAD: Spatial-Spectral Quasi-Attention Recurrent Network for Hyperspectral Image Denoising](https://ieeexplore.ieee.org/abstract/document/9732909).

## Introduction

This article presents a novel end-to-end model based on encoder–decoder architecture for hyperspectral image (HSI) denoising, named spatial-spectral quasi-attention recurrent network, denoted as SQAD. The central goal of this work is to incorporate the intrinsic properties of HSI noise to construct a practical feature extraction module while maintaining high-quality spatial and spectral information. Accordingly, we first design a spatial-spectral quasi-recurrent attention unit (QARU) to address that issue. QARU is the basic building block in our model, consisting of spatial component and spectral component, and each of them involves a two-step calculation. Remarkably, the quasi-recurrent pooling function in the spectral component could explore the relevance of spatial features in the spectral domain. The spectral attention calculation could strengthen the correlation between adjacent spectra and provide the intrinsic properties of HSI noise distribution in the spectral dimension. Apart from this, we also design a unique skip connection consisting of channelwise concatenation and transition block in our model to convey the detailed information and promote the fusion of the low-level features with the high-level ones. Such a design helps maintain better structural characteristics, and spatial and spectral fidelities when reconstructing the clean HSI. Qualitative and quantitative experiments are performed on publicly available datasets. The results demonstrate that SQAD outperforms the state-of-the-art methods of visual effect and objective evaluation metrics.

## Code Usage

1. Data Preparation

   - Download the ICVL hyperspectral dataset from [here](http://icvl.cs.bgu.ac.il/hyperspectral/).
   - Split the dataset according to `data/ICVL_train.txt` and `data/ICVL_test_*.txt`.
   - Create ICVL training dataset by `python utility/MyLMDB.py`
   - Generate your own testing dataset  by `matlab utility/matlab/make_testing_data.m`

2. Evaluation (with pretrained models)

   - Gaussian noise removal

     `python eval.py -t gauss -p sqad -r -rp checkpoints/sqad/model_gauss.pth`

   - Complex noise removal

     ``python eval.py -t gauss -p sqad -r -rp checkpoints/sqad/model_complex.pth``

## Citation

If you find our work useful in your research or publication, please cite:

```latex
@article{pan2022sqad,
  title={SQAD: Spatial-Spectral Quasi-Attention Recurrent Network for Hyperspectral Image Denoising},
  author={Pan, Erting and Ma, Yong and Mei, Xiaoguang and Fan, Fan and Huang, Jun and Ma, Jiayi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  year={2022},
  volume={60},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2022.3156646}}
```

## Acknowledgement

This repo is built mainly based on [QRNN](https://arxiv.org/abs/1611.01576), and also borrow codes from [QRNN3D](https://github.com/Vandermode/QRNN3D) . We thank a lot for their contributions to the community.