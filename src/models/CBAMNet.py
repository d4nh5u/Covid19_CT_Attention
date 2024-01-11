#CBAMNet-50

import torch
from torch import Tensor
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

from pathlib import Path
import sys
if str(Path().absolute()) not in sys.path:
    sys.path.append(str(Path().absolute()))
    

#（1）channel
class channel_attention(nn.Module):
  def __init__(self, in_channel, ratio=4):
    super(channel_attention, self).__init__()

    #[b,c,h,w]==>[b,c,1,1]
    self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    #通道數下降4倍
    self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)
    #恢復通道数
    self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
    b, c, h, w = inputs.shape

    #[b,c,h,w]==>[b,c,1,1]
    max_pool = self.max_pool(inputs)
    avg_pool = self.avg_pool(inputs)

    #[b,c,1,1]==>[b,c]
    max_pool = max_pool.view([b,c])
    avg_pool = avg_pool.view([b,c])

    #[b,c]==>[b,c//4]
    x_maxpool = self.fc1(max_pool)
    x_avgpool = self.fc1(avg_pool)

    x_maxpool = self.relu(x_maxpool)
    x_avgpool = self.relu(x_avgpool)

    #[b,c//4]==>[b,c]
    x_maxpool = self.fc2(x_maxpool)
    x_avgpool = self.fc2(x_avgpool)

    #两種池化结果相加 [b,c]==>[b,c]
    x = x_maxpool + x_avgpool
    x = self.sigmoid(x)
    #[b,c]==>[b,c,1,1]
    x = x.view([b,c,1,1])

    outputs = inputs * x

    return outputs

# ---------------------------------------------------- #
#（2）spatial
class spatial_attention(nn.Module):
  def __init__(self, kernel_size=7):
    super(spatial_attention, self).__init__()

    #為了保持卷積前后的特徵圖shape相同
    padding = kernel_size // 2
    #[b,2,h,w]==>[b,1,h,w]
    self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()

    # 前向传播
  def forward(self, inputs):

    # 在通道維度上最大池化 [b,1,h,w]  keepdim保留原有深度
    # 返回值是在某維度的最大值和對應的索引
    x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

    # 在通道維度上平均池化 [b,1,h,w]
    x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
    # 池化后的结果在通道維度上堆叠 [b,2,h,w]
    x = torch.cat([x_maxpool, x_avgpool], dim=1)

    # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
    x = self.conv(x)
    x = self.sigmoid(x)
    outputs = inputs * x

    return outputs

# ---------------------------------------------------- #
#（3）CBAM
class cbamblock(nn.Module):
  def __init__(self, in_channel, ratio=4, kernel_size=7):
    super(cbamblock, self).__init__()

    self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
    self.spatial_attention = spatial_attention(kernel_size=kernel_size)

  def forward(self, inputs):
    x = self.channel_attention(inputs)
    x = self.spatial_attention(x)
    return x

# ---------------------------------------------------- #
class CBAMResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
      super(CBAMResNet50BasicBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
      self.bn1 = nn.BatchNorm2d(outs[0])
      self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
      self.bn2 = nn.BatchNorm2d(outs[1])
      self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
      self.bn3 = nn.BatchNorm2d(outs[2])
      self.cbam = cbamblock(outs[2])

    def forward(self, x):
      out = self.conv1(x)
      out = F.relu(self.bn1(out))

      out = self.conv2(out)
      out = F.relu(self.bn2(out))

      out = self.conv3(out)
      out = self.bn3(out)

      out = self.cbam(out)

      return F.relu(out + x)


class CBAMResNet50DownBlock(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
      super(CBAMResNet50DownBlock, self).__init__()
      # out1, out2, out3 = outs
      # print(outs)
      self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
      self.bn1 = nn.BatchNorm2d(outs[0])
      self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
      self.bn2 = nn.BatchNorm2d(outs[1])
      self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
      self.bn3 = nn.BatchNorm2d(outs[2])
      self.cbam = cbamblock(outs[2])

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

      out = self.cbam(out)

      return F.relu(x_shortcut + out)


class CBAMResNet50(nn.Module):
    def __init__(self):
      super(CBAMResNet50, self).__init__()
      self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = nn.Sequential(
        CBAMResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
      )

      self.layer2 = nn.Sequential(
        CBAMResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
      )

      self.layer3 = nn.Sequential(
        CBAMResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
      )

      self.layer4 = nn.Sequential(
        CBAMResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        CBAMResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
      )

      self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

      self.fc = nn.Linear(2048, 2)  #check

      self.name = "CBAMRetNet-50"

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
    
    model = CBAMResNet50()
    model.to('cpu')
    summary(model, input_size=(1, 3, 224, 224))