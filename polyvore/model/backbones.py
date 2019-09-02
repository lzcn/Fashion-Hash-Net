import logging
import math

import torch
import torch.nn as nn

from torchvision import models

LOGGER = logging.getLogger(__name__)


# TODO: use more elegant implementation for backbone
class AlexNet(nn.Module):
    """AlexNet backbone as the feature extractor."""

    def __init__(self):
        """Deep content from AlexNet."""
        super().__init__()
        self.dim = 4096
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # replace the FCs to Conv2d
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(256, 4096, 6),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        """Forward."""
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.dim)

    def init_weights(self):
        pass

    def load_pretrained(self, state_dict):
        own_state = self.state_dict()
        for key, param in state_dict.items():
            if key in own_state:
                own_param = own_state[key]
                if param.shape == own_state[key].shape:
                    own_param.copy_(param)
                else:
                    # for layers in classifier part
                    own_param.copy_(param.view_as(own_param))
            else:
                LOGGER.info("Weight %s is dropped.", key)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, num_group=32, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_group, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_group, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, num_group=32, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_group, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_group, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(num_group, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, num_group=32, tailed=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_group, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group=num_group)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, num_group=num_group
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, num_group=num_group
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, num_group=num_group
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.tailed = tailed
        self.dim = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.gn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.gn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, num_group=32):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(num_group, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, num_group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.tailed:
            return x
        x = self.fc(x)

        return x

    def init_weights(self):
        pass

    def load_pretrained(self, state_dict):
        own_state = self.state_dict()
        for key, param in state_dict.items():
            if key in own_state:
                own_param = own_state[key]
                if param.shape == own_state[key].shape:
                    own_param.copy_(param)
                else:
                    own_param.copy_(param.resize_as_(own_param))
            else:
                LOGGER.info("Weight %s is dropped.", key)


def alexnet(pretrained=True, **kwargs):
    model = AlexNet()
    if pretrained:
        state_dict = models.alexnet(pretrained=True).state_dict()
        model.load_pretrained(state_dict)
    return model


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = torch.load("weights/resnet18.tar")
        model.load_pretrained(state_dict)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.load("weights/resnet34.tar")
        model.load_pretrained(state_dict)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.load("weights/resnet50.tar")
        model.load_pretrained(state_dict)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = torch.load("weights/resnet101.tar")
        model.load_pretrained(state_dict)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = torch.load("weights/resnet152.tar")
        model.load_pretrained(state_dict)
    return model
