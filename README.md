# SNGAN.pytorch
An unofficial Pytorch implementation of [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-). 
For official Chainer implementation please refer to [https://github.com/pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection)

Our implementation achieves Inception score of **7.90** and FID score of **20.74**, on CIFAR-10 unconditional image generation task.
In comparison, the original paper achieves **8.22** and **21.7** respectively.

NOTE: Currently, this repo only supports unconditional version of CIFAR-10 image generation task.

## Setup

### install libraries:
`pip install -r requirements.txt`

### prepare fid statistic file
 ```bash
mkdir fid_stat
```
Download the pre-calculated statistics for CIFAR10, 
[fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz), to `./fid_stat`.

### train
```bash
sh exps/sngan_cifar10.sh
```

## About Inception score and FID score
We use the official implementation scripts to calculate 
Inception score ([https://github.com/openai/improved-gan/tree/master/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score)) 
and FID score ([https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)).
