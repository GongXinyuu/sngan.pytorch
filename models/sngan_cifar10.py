import torch.nn as nn
from .gen_resblock import GenBlock
from .dis_reblock import OptimizedDisBlock, DisBlock


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

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
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
