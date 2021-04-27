import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .trainner import  trainner

class AlexNet(nn.Module ,trainner):
  
  def __init__(self):
    super().__init__()
    self.pool_size = 3
    self.conv11_1 = nn.Conv2d(3 , 96 , 11 , stride=4 , padding=0) #224x224x3 ==> 54x54x96

    self.avgpool1 = nn.MaxPool2d(self.pool_size , stride = 2 , padding=0) #54x54x96 ==> 26x26x96

    self.conv5_1 = nn.Conv2d(96 , 256 , 5 , stride=1 , padding=2) #26x26x96 ==> 26x26x256

    self.avgpool2 = nn.MaxPool2d(self.pool_size , stride = 2 , padding=0) #26x26x256 ==> 12x12x256

    self.conv3_1 = nn.Conv2d(256 , 384 , 3 , stride=1 , padding=1) #12x12x256 ==> 12x12x384
    self.conv3_2 = nn.Conv2d(384 , 384 , 3 , stride=1 , padding=1) #12x12x384 ==> 12x12x384

    self.conv3_3 = nn.Conv2d(384 , 256 , 3 , stride=1 , padding=1) #12x13x384 ==> 12x12x256

    self.avgpool3 = nn.MaxPool2d(self.pool_size , stride = 2 , padding=0) #12x12x256 ==> 5x5x256

    self.fc1 = nn.Linear(5*5*256,4096)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(4096 , 4096)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(4096 , 10)
      
  def forward(self , x):
    x= F.relu(self.conv11_1(x))
    x= self.avgpool1(x)

    x= F.relu(self.conv5_1(x))
    x= self.avgpool2(x)
    
    x= F.relu(self.conv3_1(x))
    x= F.relu(self.conv3_2(x))
    x= F.relu(self.conv3_3(x))
    x= self.avgpool3(x)

    x= x.view(x.shape[0],-1)

    x= F.relu(self.fc1(x))
    x= self.dropout1(x)
    x= F.relu(self.fc2(x))
    x= self.dropout2(x)
    x= self.fc3(x)

    return x
