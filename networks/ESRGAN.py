import torch
import torch.nn as nn
import functools
import torch.nn.init as init
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Upsample

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                #nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                #nn.LeakyReLU(),
                #nn.PixelShuffle(upscale_factor=2),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(filters,filters,kernel_size=3, stride=1, padding=1)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=24):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
#        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

#        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x 

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=24, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + identity

        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growths)
        self.rdb2 = ResidualDenseBlock(channels, growths)
        self.rdb3 = ResidualDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = out * 0.2 + identity

        return out
class ESRGAN(nn.Module):
    def __init__(self,chan_in=3,chan_out=3,scale=4,nRRDB=16) -> None:
        super(ESRGAN, self).__init__()
        # The first layer of convolutional layer.
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.scale = scale
        self.nRRDB = nRRDB
        self.conv_block1 = nn.Conv2d(self.chan_in, 32, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(self.nRRDB):
            trunk += [ResidualResidualDenseBlock(32, 16)]
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv_block2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv_block4 = nn.Conv2d(32, self.chan_out, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        if not self.scale == 1:
            out = self.upsampling(F.interpolate(out, scale_factor=self.scale, mode="bilinear"))
        # out = self.upsampling(F.interpolate(out, scale_factor=2, mode="bilinear"))
        out = self.conv_block3(out)
        out = self.conv_block4(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1