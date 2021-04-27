import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .trainner import  trainner

class ResNet18(nn.Module , trianner):
  def __init__(self):
    super().__init__()
    self.conv7x7 = nn.Conv2d(3 , 64 , 7 , stride=2 , padding=3) #3x224x224 ==> 64x112x122
    self.pool = nn.MaxPool2d(2 , stride=2) #64x112x112  ==>  64x56x56

    self.block1 = nn.Sequential(ResBlock(64 , 64 , 2),
                                ResBlock(64 , 64 , 2)
                                )                         #64x56x56 ==> 64x56x56
    
    self.block2 = nn.Sequential(ResBlock(64 , 128 , 2 , stride=2 , increase_channels=True) ,
                                ResBlock(128 , 128 , 2) ,
                                )                         #128x28x28 
        
    self.block3 = nn.Sequential(ResBlock(128 , 256 , 2 , stride=2 , increase_channels=True) ,
                                ResBlock(256 , 256 , 2) ,
                                )                         #256x14x14         
    
    self.block4 = nn.Sequential(ResBlock(256 , 512 , 2 , stride=2 , increase_channels=True) ,
                                ResBlock(512 , 512 , 2),  #512x7x7
                                nn.AvgPool2d(7) ,
                                nn.Flatten() ,
                                nn.Linear(512 , 10)
                                )     

  def forward(self , x):
    x=self.pool(   F.relu(self.conv7x7(x))  )                  
            
            
class ResBlock(nn.Module):
  def __init__(self , in_channels , out_channels , num_convs, stride=1 , increase_channels=False):
    #if increase_channels == False ==> in_channels = out_channels
    super().__init__()
    self.increase_channels = increase_channels
    layers = []
    layers.append(nn.Conv2d(in_channels , out_channels , 3 , stride = stride, padding =1))
    layers.append(nn.BatchNorm2d(out_channels))
    for i in range(num_convs-1):
      layers.append(nn.ReLU())
      layers.append(nn.Conv2d(out_channels , out_channels , 3 , stride =1 , padding =1))
      layers.append(nn.BatchNorm2d(out_channels))
    self.res_block = nn.Sequential(*layers)
    if self.increase_channels:
      self.conv1x1 = nn.Conv2d(in_channels , out_channels , 1 , stride=stride , padding = 0)

  def forward(self , x ):
    if self.increase_channels:
        x = self.res_block(x) + self.conv1x1(x)
    else:
        x = self.res_block(x) + x
    return F.relu(x)
