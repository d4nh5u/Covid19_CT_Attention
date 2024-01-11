# SE-ResNet-50

import torch
from torch import Tensor
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

from pathlib import Path
import sys
if str(Path().absolute()) not in sys.path:
    sys.path.append(str(Path().absolute()))

class SEBlock(nn.Module):
  def __init__(self, in_channels, reduction=16):
    super(SEBlock, self).__init__()
    self.se = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Conv2d(in_channels, in_channels // 16, kernel_size=1), #reduction=16
      nn.ReLU(),
      nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
      nn.Sigmoid()
    )

  def forward(self, x):
    y = self.se(x)
    return x * y


class SEResNet50BasicBlock(nn.Module):
  def __init__(self, in_channel, outs, kernerl_size, stride, padding):
    super(SEResNet50BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
    self.bn1 = nn.BatchNorm2d(outs[0])
    self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
    self.bn2 = nn.BatchNorm2d(outs[1])
    self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
    self.bn3 = nn.BatchNorm2d(outs[2])
    self.se = SEBlock(outs[2])

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(self.bn1(out))

    out = self.conv2(out)
    out = F.relu(self.bn2(out))

    out = self.conv3(out)
    out = self.bn3(out)

    out = self.se(out)

    return F.relu(out + x)


class SEResNet50DownBlock(nn.Module):
  def __init__(self, in_channel, outs, kernel_size, stride, padding):
    super(SEResNet50DownBlock, self).__init__()
    # out1, out2, out3 = outs
    # print(outs)
    self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
    self.bn1 = nn.BatchNorm2d(outs[0])
    self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
    self.bn2 = nn.BatchNorm2d(outs[1])
    self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
    self.bn3 = nn.BatchNorm2d(outs[2])
    self.se = SEBlock(outs[2])

    self.extra = nn.Sequential(
      nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
      nn.BatchNorm2d(outs[2])
    )

  def forward(self, x):
    x_shortcut = self.extra(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    out = self.se(out)

    return F.relu(x_shortcut + out)


class SEResNet50(nn.Module):
  def __init__(self):
    super(SEResNet50, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = nn.Sequential(
      SEResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
    )

    self.layer2 = nn.Sequential(
      SEResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
      SEResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50DownBlock(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
    )

    self.layer3 = nn.Sequential(
      SEResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
      SEResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
    )

    self.layer4 = nn.Sequential(
      SEResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
      SEResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      SEResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
    )

    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    self.fc = nn.Linear(2048, 2)  #check

    self.name = "SERetNet-50"

  def forward(self, x):
    out = self.conv1(x)
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = out.reshape(x.shape[0], -1)
    out = self.fc(out)
    return out
  
if __name__ == '__main__':
    project_path = Path().absolute()
    print('Project Path:', project_path)
    
    model = SEResNet50()
    model.to('cpu')
    summary(model, input_size=(1, 3, 224, 224))