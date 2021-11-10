import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .trainner import  trainner

class Inception(nn.Module , trianner):
  def __init__(self):
    super().__init__()
    pool_stride=2
    pool_size=3
    pool_padding=1

    in_ib1=(192,256)
    ib1=((64, (96, 128), (16, 32), 32),
         (128, (128, 192), (32, 96), 64))

    in_ib2=(480,512,512,512,528)
    ib2=(
      (192, (96, 208), (16, 48), 64),
      (160, (112, 224), (24, 64), 64),
      (128, (128, 256), (24, 64), 64),
      (112, (144, 288), (32, 64), 64),
      (256, (160, 320), (32, 128), 128))

    in_ib3=(832,832)
    ib3=(
    (256, (160, 320), (32, 128), 128),
    (384, (192, 384), (48, 128), 128)
    )
    self.block1=nn.Sequential(nn.Conv2d(3 , 64 , 7 , stride=2 , padding=3) , nn.ReLU() , nn.MaxPool2d(pool_size , pool_stride , pool_padding))

    self.block2=nn.Sequential(nn.Conv2d(64 , 64 , 1) , nn.ReLU() ,
                              nn.Conv2d(64 , 192 , 3 , stride=1 , padding=1) , nn.ReLU() , nn.MaxPool2d(pool_size , pool_stride , pool_padding))
    
    self.block3=nn.Sequential(inception_block(in_ib1[0] , ib1[0]) , 
                              inception_block(in_ib1[1] , ib1[1]) , nn.MaxPool2d(pool_size , pool_stride , pool_padding))
    
    self.block4=nn.Sequential(inception_block(in_ib2[0] , ib2[0]) , 
                              inception_block(in_ib2[1] , ib2[1]) , 
                              inception_block(in_ib2[2] , ib2[2]) , 
                              inception_block(in_ib2[3] , ib2[3]) , 
                              inception_block(in_ib2[4] , ib2[4]) , nn.MaxPool2d(pool_size , pool_stride , pool_padding) )
    
    self.block5=nn.Sequential(inception_block(in_ib3[0] , ib3[0]) ,
                              inception_block(in_ib3[1] , ib3[1]) , nn.AvgPool2d(7) , nn.Flatten())
    self.fc = nn.Linear(1024 , 10)

  def forward(self ,x ):
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.block4(x)
    x=self.block5(x)
    x=self.fc(x)


class inception_block(nn.Module): 

  def __init__(self , in_channels  , c):
    super().__init__()
    self.path_conv1 = nn.Sequential ( nn.Conv2d(in_channels , c[0] , 1  , 1 , 0) , nn.ReLU())
    self.path_conv3 = nn.Sequential ( nn.Conv2d(in_channels , c[1][0] , 1 , 1 , 0 ) , nn.ReLU() , nn.Conv2d(c[1][0] , c[1][1] , 3 , 1 , 1) , nn.ReLU())
    self.path_conv5 = nn.Sequential ( nn.Conv2d(in_channels , c[2][0] , 1 , 1 , 0) , nn.ReLU() , nn.Conv2d(c[2][0] , c[2][1] , 5 , 1 , 2), nn.ReLU())
    self.path_pool = nn.Sequential ( nn.MaxPool2d(3,1,1) , nn.Conv2d(in_channels , c[3] , 1 , 1 , 0) , nn.ReLU() )

  def forward(self ,x ):
    p1 = self.path_conv1(x)
    p2 = self.path_conv3(x)
    p3 = self.path_conv5(x)
    p4 = self.path_pool(x)

    return  torch.cat(( p1,p2,p3,p4) ,  dim=1) 