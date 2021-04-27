In this simple lib I implemented some of the famous cnn architectures in pytorch like VGG16 , ResNet , InceptionV1 and Alexnet

You use the lib to to train any of these models on your custom data using few lines of code

First prepare your pytirch dataloader
 ```python
import troch.nn as nn
from torchvision.datasets import CIFAR10 
from torchvision import transforms
from torch.utils.data import DataLoader
mytransforms = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize(224),
                   transforms.Normalize((0,0,0),(255.0 ,255.0 , 255.0)) 
                               ])
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=mytransforms)
cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=mytransforms

train_loader = DataLoader(cifar10_train , batch_size=64 , shuffle=True)
test_loader = DataLoader(cifar10_test , batch_size=64)

num_classes=10

```
Then create the model ,  optimizer and define the loss function

```
from cnnModels.Alexnet import AlexNet
alexnet = AlexNet(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=1e-4)

```
Now train the model you want using just one line of code

```
alexnet.Train(device ,criterion, train_loader, test_loader , optimizer, num_epochs=10)
```
