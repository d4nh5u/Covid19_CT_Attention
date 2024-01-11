import torch
from torch import Tensor
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

class VGG16(nn.Module):
  def __init__(self, num_classes=2):
    super(VGG16, self).__init__()
    self.features = nn.Sequential(
      #1
      nn.Conv2d(3,64,kernel_size=3,padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      #2
      nn.Conv2d(64,64,kernel_size=3,padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      #3
      nn.Conv2d(64,128,kernel_size=3,padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      #4
      nn.Conv2d(128,128,kernel_size=3,padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      #5
      nn.Conv2d(128,256,kernel_size=3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      #6
      nn.Conv2d(256,256,kernel_size=3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      #7
      nn.Conv2d(256,256,kernel_size=3,padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      #8
      nn.Conv2d(256,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      #9
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      #10
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      #11
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      #12
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      #13
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      nn.AvgPool2d(kernel_size=1,stride=1),
    )
    self.classifier = nn.Sequential(
      #14
      nn.Linear(512*7*7,4096),
      nn.ReLU(True),
      nn.Dropout(),
      #15
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      #16
      nn.Linear(4096,num_classes),
    )

        #self.classifier = nn.Linear(512, 10)
    self.name = "VGG-16"

  def forward(self, x):
    out = self.features(x)
    #print(out.shape)
    out = out.view(out.size(0), -1)
    #print(out.shape)
    out = self.classifier(out)
    #print(out.shape)
    return out


if __name__ == '__main__':

    model = VGG16()
    model.to('cpu')
    summary(model, input_size=(1, 3, 224, 224))