"""
Implementation of "Fully Convolutional Networks for Continuous Sign Language Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = self.norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MainStream(nn.Module):
    def __init__(self, vocab_size):
        super(MainStream, self).__init__()

        # cnn
        # first layer: channel 3 -> 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # 4 basic blocks
        channels = [32, 64, 128, 256, 512]
        layers = []
        for num_layer in range(len(channels) - 1):
            layers.append(BasicBlock(channels[num_layer], channels[num_layer + 1]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # encoder G1, two F5-S1-P2-M2
        self.enc1_conv1 = nn.Conv1d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.enc1_bn1 = nn.BatchNorm1d(512)
        self.enc1_ln1 = nn.LayerNorm(512)
        self.enc1_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc1_conv2 = nn.Conv1d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.enc1_bn2 = nn.BatchNorm1d(512)
        # self.enc1_ln2 = nn.LayerNorm(512)
        self.enc1_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.enc1 = nn.Sequential(self.enc1_conv1, self.enc1_bn1, self.relu, self.enc1_pool1,
        #                          self.enc1_conv2, self.enc1_bn2, self.relu, self.enc1_pool2)

        # encoder G2, one F3-S1-P1
        self.enc2_conv = nn.Conv1d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.enc2_bn = nn.BatchNorm1d(1024)
        # self.enc2_ln = nn.LayerNorm(1024)
        # self.enc2 = nn.Sequential(self.enc2_conv, self.enc2_bn, self.relu)
        # self.act_tanh = nn.Tanh()
        self.fc = nn.Linear(1024, vocab_size)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, video, len_video=None):
        """
        x: [batch, num_f, 3, h, w]
        """
        # print("input: ", video.size())
        bs, num_f, c, h, w = video.size()

        x = video.reshape(-1, c, h, w)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layers(x)
        x = self.avgpool(x).squeeze_()
        x = x.reshape(bs, 512, -1)  # [bs, 512, t]

        x = self.enc1_conv1(x) # [bs, 512, t/2]
        x = self.enc1_bn1(x)
        x = self.relu(x)
        x = self.enc1_pool1(x)  # [bs, 512, t/2]

        x = self.enc1_conv2(x)  # [bs, 512, t/2]
        x = self.enc1_bn2(x)
        x = self.relu(x)
        x = self.enc1_pool2(x)  # [bs, 512, t/4]


        # enc2
        x = self.enc2_conv(x)
        out = self.relu(x)
        # out = self.act_tanh(x)

        out = out.permute(0, 2, 1)
        logits = self.fc(out)
        len_video = torch.Tensor(bs * [logits.size(1)]).to(logits.device)
        return logits, len_video


if __name__ == "__main__":
    x = torch.randn(1, 30, 3, 112, 112)
    model = MainStream(1233)
    for i in range(500):
        out = model(x)
        print(out[0].shape, out[1])