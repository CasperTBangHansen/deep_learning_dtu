#!/usr/bin/env python
"""
ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torchvision # type: ignore
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


class Block(ABC, nn.Module):
    """ Building block for Resnet"""
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """ Init building block"""
        super(Block, self).__init__()

    @property
    @abstractmethod
    def expansion(self) -> int:
        """ Each parent class should have an expansion constant"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Method for doing the forward pass"""

block_T = Callable[[int, int, int], Block]


class BasicBlock(Block):
    """ Basic building block for Resnet"""
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__(in_planes, planes, stride)

        # Create the first layer
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Create the second layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Activation function
        self.relu = nn.ReLU()

        # Create a shortcut if condition is meet
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Method for doing the forward pass"""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(Block):
    """ Bottleneck building block for Resnet"""
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__(in_planes, planes, stride)

        # Create the first layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Create the second layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Create the third layer
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Activation function
        self.relu = nn.ReLU()

        # Create a shortcut if condition is meet
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Method for doing the forward pass"""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """ Class for create a ResNet model"""
    def __init__(self, block: block_T, channels: int, num_blocks: List[int], num_classes: int = 10):

        # Init
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.channels = channels

        # Creates the initial layer 
        self.conv1 = nn.Conv2d(
            self.channels,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        # The following layers are the classical 4 layers of ResNets
        self.layer1, _ = self.__make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2, _ = self.__make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3, _ = self.__make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4, expansion = self.__make_layer(block, 512, num_blocks[3], stride=2)

        # Creates the output layers (average pooling and linear layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)

    def __make_layer(self, block: block_T, planes: int, num_blocks: int, stride: int) -> Tuple[nn.Sequential, int]:
        """ Creates the classical Resnet layers with different amount of blocks"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Create block
            init_block = block(self.in_planes, planes, stride)
            layers.append(init_block)
            # Updates in_planes for the next loop/functioncall
            self.in_planes = planes * init_block.expansion

        # Combines the blocks into a sequential block
        return nn.Sequential(*layers), init_block.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Method for doing the forward pass"""
        # Initial block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # The 4 layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Output block
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Predefined ResNets
def ResNet18(channels: int = 3, num_classes: int = 10) -> ResNet:
    """ Creastes a ResNet18"""
    return ResNet(BasicBlock, channels, [2, 2, 2, 2], num_classes)

def ResNet34(channels: int = 3, num_classes: int = 10) -> ResNet:
    """ Creastes a ResNet34"""
    return ResNet(BasicBlock, channels, [3, 4, 6, 3], num_classes)

def ResNet50(channels: int = 3, num_classes: int = 10) -> ResNet:
    """ Creastes a ResNet50"""
    return ResNet(Bottleneck, channels, [3, 4, 6, 3], num_classes)

def ResNet101(channels: int = 3, num_classes: int = 10) -> ResNet:
    """ Creastes a ResNet101"""
    return ResNet(Bottleneck, channels, [3, 4, 23, 3], num_classes)

def ResNet152(channels: int = 3, num_classes: int = 10) -> ResNet:
    """ Creastes a ResNet152"""
    return ResNet(Bottleneck, channels, [3, 8, 36, 3], num_classes)

# Tests
def test18():
    """
    Checks if our implementation of ResNet18
    is the same as Pytorch implementation of ResNet18
    """
    # Init our ResNet18 and Pytorch ResNet18
    net = ResNet18(num_classes=1000)
    resnet18 = torchvision.models.resnet18(pretrained=False)
    # Random input
    x = torch.randn(1, 3, 224, 224)
    # Get output
    y_resnet18 = resnet18(x)
    y_net = net(x)

    # Print sizes
    print("ResNet18:")
    print('\tOur resnet:')
    print(f'\tThe output is of size {y_net.size()}')
    print('\tPytorch resnet:')
    print(f'\tThe output is of size {y_resnet18.size()}')

    # Print architecture
    # print('Our resnet:')
    # print(net)
    # print('Pytorch resnet:')
    # print(resnet18)

def test50():
    """
    Checks if our implementation of ResNet50
    is the same as Pytorch implementation of ResNet50
    """
    # Init our ResNet50 and Pytorch ResNet50
    net = ResNet50(num_classes=1000)
    resnet50 = torchvision.models.resnet50(pretrained=False)
    # Random input
    x = torch.randn(1, 3, 224, 224)
    # Get output
    y_resnet50 = resnet50(x)
    y_net = net(x)

    # Print sizes
    print("ResNet50:")
    print('\tOwn resnet:')
    print(f'\tThe output is of size {y_net.size()}')
    print('\tPytorch resnet:')
    print(f'\tThe output is of size {y_resnet50.size()}')
    
    # Print architecture
    # print('Our resnet:')
    # print(net)
    # print('Pytorch resnet:')
    # print(resnet50)

if __name__ == '__main__':
    # Run tests
    test18()
    test50()