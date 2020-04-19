import torch.nn as nn


# from base_backbone import BaseBackbone
# else:
#     from .base_backbone import BaseBackbone
__all__ = ['MobileFaceNet']


class Flatten(nn.Module):
    @staticmethod
    def forward(input):
        return input.view(input.size(0), -1)


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# class MobileFaceNet(BaseBackbone):
class MobileFaceNet(nn.Module):
    def __init__(self, ratio=1, layers=None, pool=True, input_shape=None):
        super(MobileFaceNet, self).__init__()
        if layers is None:
            layers = [2, 8, 16, 4]
        self.input_shape = input_shape
        self.layers = layers
        self.pool = pool

        self.conv1 = Conv_block(3, 64 * ratio, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.conv2_dw = Conv_block(64*ratio, 64*ratio, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64*ratio)
        self.conv2 = Residual(64 * ratio, num_block=self.layers[0], groups=64 * ratio, kernel=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.conv_23 = Depth_Wise(64 * ratio, 64 * ratio, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                  groups=128 * ratio)
        self.conv_3 = Residual(64 * ratio, num_block=self.layers[1], groups=128 * ratio, kernel=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.conv_34 = Depth_Wise(64 * ratio, 128 * ratio, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128 * ratio, num_block=self.layers[2], groups=256, kernel=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.conv_45 = Depth_Wise(128 * ratio, 128 * ratio, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                  groups=512 * ratio)
        self.conv_5 = Residual(128 * ratio, num_block=self.layers[3], groups=256, kernel=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.feature_shape = self.input_shape[1] // 16
        if self.pool:
            self.conv_6_sep = Conv_block(128 * ratio, 512 * ratio, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            self.conv_6_dw = Linear_block(512 * ratio, 512 * ratio, groups=512 * ratio,
                                          kernel=(self.feature_shape, self.feature_shape),
                                          stride=(1, 1), padding=(0, 0))
            self.output_shape = [512 * ratio, 1, 1]
        else:
            self.conv_6_sep = Linear_block(128 * ratio, 512 * ratio, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            self.output_shape = [512 * ratio, self.feature_shape, self.feature_shape]
        self._init_params()

    def get_output_shape(self):
        return self.output_shape

    def forward(self, input):
        output = {}
        x = input['images']
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        if self.pool:
            x = self.conv_6_dw(x)
        x = x.view(x.size(0), -1)
        output.update({'feature': x})
        return output


def mobilefacenet(**kwargs):
    model = MobileFaceNet(**kwargs)
    return model

if __name__ == '__main__':
    model = mobilefacenet()
    print('mobilefacenet')