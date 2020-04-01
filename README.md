# SNGAN.pytorch
An unofficial Pytorch implementation of [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-). 
For official Chainer implementation please refer to [https://github.com/pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection)

Our implementation achieves Inception score of **8.21** and FID score of **14.21** on unconditional CIFAR-10 image generation task.
In comparison, the original paper claims **8.22** and **21.7** respectively.

## Set-up

### install libraries:
```bash
pip install -r requirements.txt
```

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

### test
```bash
mkdir pre_trained
```
Download the pre-trained SNGAN model [sngan_cifar10.pth](https://drive.google.com/file/d/1koEJbx9anP2-BEMrqX6jgWXAvEUXG0AU/view?usp=sharing) to `./pre_trained`.
Run the following script:
```bash
sh exps/eval.sh
```

## Acknowledgement

1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. The code of Spectral Norm GAN is inspired by [https://github.com/pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection) (official).
