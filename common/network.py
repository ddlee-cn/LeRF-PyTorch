import torch
import torch.nn as nn
import torch.nn.functional as F


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))


class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class DenseConv(nn.Module):
    def __init__(self, in_nf, nf=64, k=1, p=0):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, k, padding=p)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out

class SRUnit(nn.Module):
    def __init__(self, mode, nf, upscale, outC):
        super(SRUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, outC * upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x


class SRNet(nn.Module):
    def __init__(self, mode, nf=64, upscale=None, outC=1):
        super(SRNet, self).__init__()
        self.mode = mode

        if 'x1' in mode:
            assert upscale == None
        if mode == 'Sx1':
            self.model = SRUnit('2x2', nf, upscale=1, outC=outC)
            self.K = 2
            self.S = 1
        elif mode == 'SxN':
            self.model = SRUnit('2x2', nf, upscale=upscale, outC=outC)
            self.K = 2
            self.S = upscale
        elif mode == 'Dx1':
            self.model = SRUnit('2x2d', nf, upscale=1, outC=outC)
            self.K = 3
            self.S = 1
        elif mode == 'DxN':
            self.model = SRUnit('2x2d', nf, upscale=upscale, outC=outC)
            self.K = 3
            self.S = upscale
        elif mode == 'Yx1':
            self.model = SRUnit('1x4', nf, upscale=1, outC=outC)
            self.K = 3
            self.S = 1
        elif mode == 'YxN':
            self.model = SRUnit('1x4', nf, upscale=upscale, outC=outC)
            self.K = 3
            self.S = upscale
        elif mode == 'Cx1':
            self.model = SRUnit('1x4', nf, upscale=1, outC=outC)
            self.K = 4
            self.S = 1
        elif mode == 'CxN':
            self.model = SRUnit('1x4', nf, upscale=upscale, outC=outC)
            self.K = 4
            self.S = upscale
        elif mode == 'Tx1':
            self.model = SRUnit('1x4', nf, upscale=1, outC=outC)
            self.K = 4
            self.S = 1
        elif mode == 'TxN':
            self.model = SRUnit('1x4', nf, upscale=upscale, outC=outC)
            self.K = 4
            self.S = upscale
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'C' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1],
                           x[:, :, 0, 2], x[:, :, 0, 3]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'T' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 2, 2], x[:, :, 3, 3]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)
        # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -
        1)  # B,C,K*K,L
        # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B, -1, (H - self.P) * (W - self.P)
                      )  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x
