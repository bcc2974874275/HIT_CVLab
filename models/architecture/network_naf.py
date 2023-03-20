import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, c):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAF(nn.Module):
    def __init__(self, in_channel, dim=64, out_channel=None, scale=4, block_num=16):
        super(NAF, self).__init__()
        if out_channel is None: out_channel = in_channel

        self.first_conv = nn.Conv2d(in_channel + 1, dim, 3, 1, 1)

        self.body = nn.Sequential(*[NAFBlock(dim, 2, 2) for _ in range(block_num)])

        self.last_layers = nn.Sequential(
            # nn.Conv2d(in_channels=dim, out_channels=(scale ** 2) * dim, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, True),
            # nn.PixelShuffle(scale),
            nn.Conv2d(in_channels=dim, out_channels=out_channel, kernel_size=3,padding=1)
        )

    def forward(self, ms, pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        x = torch.cat([ms, pan], dim=1)
        y = self.first_conv(x)
        y = self.body(y)

        y = self.last_layers(y)
        out = y + ms
        return out

    def test(self, device='cuda'):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        input_ms = torch.rand(1, 8, 64, 64)
        input_pan = torch.rand(1, 1, 256, 256)

        import torchsummaryX
        torchsummaryX.summary(self, input_ms.to(device), input_pan.to(device))


if __name__ == '__main__':
    model = NAF(8, 64, 8, 4, 16).cuda()
    ms = torch.rand([1, 8, 16, 16]).cuda()
    pan = torch.rand([1, 1, 64, 64]).cuda()
    sr = model(ms, pan)
    print(sr.shape)
    model.test()



