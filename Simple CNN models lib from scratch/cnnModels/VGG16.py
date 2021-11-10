import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .trainner import  trainner

class VGG16(nn.Module , trianner):
  def __init__(self):
    super().__init__()
    kernel_size=3
    stride=1
    padding=1
    pool_size=2
    pool_stride=2
    self.conv_block1 = self.conv_block( 3 , 64 , kernel_size , stride , padding , num_layers = 2) 
    self.maxpool1 = nn.MaxPool2d(pool_size , pool_stride)     #224==>112
    self.conv_block2 = self.conv_block( 64 , 128 , kernel_size , stride , padding ,num_layers = 2) 
    self.maxpool2 = nn.MaxPool2d(pool_size , pool_stride)     #112==>56
    self.conv_block3 = self.conv_block( 128 , 256 , kernel_size , stride , padding , num_layers = 3) 
    self.maxpool3 = nn.MaxPool2d(pool_size , pool_stride)     #56==>28
    self.conv_block4 = self.conv_block( 256 , 512 , kernel_size , stride , padding , num_layers = 3) 
    self.maxpool4 = nn.MaxPool2d(pool_size , pool_stride)     #28==>14
    self.conv_block5 = self.conv_block( 512 , 512 , kernel_size , stride , padding , num_layers = 3) 
    self.maxpool5 = nn.MaxPool2d(pool_size , pool_stride)     #14==>7
    self.fc1 = nn.Linear(7*7*512 , 4096)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear( 4096, 4096)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear( 4096 , 10 )


  def forward(self, x):
    x=self.conv_block1(x)
    x=self.maxpool1(x)
    x=self.conv_block2(x)
    x=self.maxpool2(x)
    x=self.conv_block3(x)
    x=self.maxpool3(x)
    x=self.conv_block4(x)
    x=self.maxpool4(x)
    x=self.conv_block5(x)
    x=self.maxpool5(x)
    x=x.view(x.shape[0],-1)
    x= F.relu(self.fc1(x))
    x= self.dropout1(x)
    x= F.relu(self.fc2(x))
    x= self.dropout2(x)
    x= self.fc3(x)
    return x


  def conv_block(self , in_channels , out_channels , kernel_size  , stride , padding , num_layers):
    conv_layers=[]
    for l in range(num_layers):
      conv_layers.append(nn.Conv2d(in_channels , out_channels , kernel_size , stride , padding))
      conv_layers.append(nn.ReLU())

      in_channels = out_channels

    return nn.Sequential(*conv_layers)
