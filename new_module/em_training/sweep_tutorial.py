import wandb
import torch
import torch.nn as nn

# reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L25

## build a CNN (ResNet) to classify an Tiny ImageNet image.
## run a hyperparameter sweep.

# define the CNN model
## why set bias=False for conv layers?
class BuildingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = None
        if (input_channels != output_channels) or (stride > 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
    
    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) 
        if self.downsample:
            out += self.downsample(identity)
        else:
            out += identity
        out = self.relu(out)
        return out
        

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2 = BuildingBlock(64, 64, 3, 2)
        self.block3 = BuildingBlock(64, 128, 3, 2)
        self.block4 = BuildingBlock(128, 256, 3, 2)
        self.block5 = BuildingBlock(256, 512, 3, 2)
        self.avgpool = nn.AvgPool2d((1,1))
        self.linear = nn.Linear(512, num_classes, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
        
# define optimizer and loss function
# max iteration 60 x 10^4 
# batch size 256
# SGD momentum 0.9 weight decay 0.0001
# learning rate 0.1 -> divided by 10 when the error plateus
# https://arxiv.org/pdf/1512.03385

# define dataset and dataloader
# https://huggingface.co/datasets/zh-plus/tiny-imagenet
# assume 224 x 224 crop (the input is 64 x 64 though..)
# 64 -> 32 -> 16 -> 8 -> 4 -> 2

# define training loop

# define evaluation loop

# wrap everything in a function

# define the sweep configuration and run the sweep

