import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nas_utils import random_choice


# 包括： conv2d,bn2d,relu,conv2d,bn2d,relu
# 输入 3 param : inplanes, outplanes ,k表示kernel size
# inplanes:
# outplanes:
class ConvBnRelu(nn.Module):

    def __init__(self, inplanes, outplanes, k):
        super(ConvBnRelu, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),

            nn.Conv2d(outplanes, outplanes, kernel_size=k, stride=1, padding=k // 2, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


# maxpooling block
# 包括 convd，bn2d，relu maxpooling2d。
class MaxPool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(MaxPool, self).__init__()

        self.op = nn.Sequential(
            # kernel=1 conv
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),

            nn.MaxPool2d(3, 1, padding=1)
        )

    def forward(self, x):
        return self.op(x)


# 基本cell单元，用ModuleList形式实现。 shadow_bn影子batch_normal
# 由基本的 （ConvBnRelu X 3 ， Maxpool X 1）X 4 ， Conv2d X 1 组成
# shadow_bn 是一个boolean值，是否使用影子批正则化。
# forward 中choice 是选择使用哪一些nodes

class Cell(nn.Module):
    def __init__(self, inplanes, outplanes, shadow_bn):
        super(Cell, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn

        self.nodes = nn.ModuleList([])
        for i in range(4):
            self.nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 1))
            self.nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 3))
            self.nodes.append(MaxPool(self.inplanes, self.outplanes))
        self.nodes.append(nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1))

        self.bn_list = nn.ModuleList([])
        if self.shadow_bn:
            for j in range(4):
                self.bn_list.append(nn.BatchNorm2d(outplanes))
        else:
            self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x, choice):
        path_ids = choice['path']  # eg.[0, 2, 3]   # 哪一个节点
        op_ids = choice['op']  # eg.[1, 1, 2]       # 节点的哪一个操作
        x_list = []
        for i, id in enumerate(path_ids):
            x_list.append(self.nodes[id * 3 + op_ids[i]](x))

        x = sum(x_list)  # 节点对于输出求和。
        out = self.nodes[-1](x)
        return F.relu(out)


# 超网类，三个参数，[初始化channels， 类数量，是否使用影子批处理]。
class SuperNetwork(nn.Module):
    def __init__(self, init_channels, classes=10, shadow_bn=True):
        super(SuperNetwork, self).__init__()
        self.init_channels = init_channels

        # stem ？？
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True)
        )
        # cell 集合
        self.cell_list = nn.ModuleList([])

        # 为什么是 9 ？？
        for i in range(9):
            # 遇到 3， 6 改变参数， init_channels X 2
            if i in [3, 6]:
                self.cell_list.append(Cell(self.init_channels, self.init_channels * 2, shadow_bn=shadow_bn))
                self.init_channels *= 2
            else:
                self.cell_list.append(Cell(self.init_channels, self.init_channels, shadow_bn=shadow_bn))

        # 用自适应2d pooling
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # classfier init_channels X classes
        self.classifier = nn.Linear(self.init_channels, classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化权重。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # conv2d 参数使用 mean =0 ， std = sqrt( 2 / kernel[0] X kernel[1]) 的正态分布来初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # bn2d 填充 weight 全部设置为1，bias设置为0
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # linear 使用 连续均匀分布来初始化
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, x, choice):
        # 不太清楚这个stem是什么
        x = self.stem(x)
        # 遍历 9 layer
        for i in range(9):
            x = self.cell_list[i](x, choice)
            if i in [2, 5]:
                x = nn.MaxPool2d(2, 2, padding=0)(x)
        x = self.global_pooling(x)
        x = x.view(-1, self.init_channels)
        out = self.classifier(x)

        return out


if __name__ == '__main__':
    # ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    # choice = {'path': [0, 1, 2],  # a list of shape (4, )
    #           'op': [0, 0, 0]}  # possible shapes: (), (1, ), (2, ), (3, )
    choice = random_choice(3)
    print(choice)
    layer = nn.Conv2d(12, 24, 2)
    print(layer.weight[1, 1, 1])
    nn.init.normal_(layer.weight, 0, math.sqrt(2.0 / 4))
    print(layer.weight[1, 1, 1])
    # model = SuperNetwork(init_channels=128)
    # input = torch.randn((1, 3, 32, 32))
    # print(model(input, choice))
