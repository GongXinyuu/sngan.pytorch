import torch.nn as nn
from .gen_resblock import GenBlock
from .dis_reblock import OptimizedDisBlock, DisBlock


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = 512
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(512, 256, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(256, 128, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(128, 64, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(64)
        self.c5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, 64)
        self.block2 = DisBlock(args, 64, 128, activation=activation, downsample=True)
        self.block3 = DisBlock(args, 128, 256, activation=activation, downsample=True)
        self.block4 = DisBlock(args, 256, 512, activation=activation, downsample=True)
        self.block5 = DisBlock(args, 512, 1024, activation=activation, downsample=False)

        self.l6 = nn.Linear(1024, 1, bias=False)
        if args.d_spectral_norm:
            self.l6 = nn.utils.spectral_norm(self.l6)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l6(h)

        return output
