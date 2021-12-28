#!/usr/bin/env python
"""
SuperArchitecture in PyTorch.

Reference:
[1] M. F. Kasim, D. Watson-Parris, L. Deaconu, S. Oliver,
    P. Hatfield, D. H. Froula, G. Gregori, M. Jarvis, S. Khatiwala,
    J. Korenaga, J. Topp-Mugglestone, E. Viezzer, S. M. Vinko
    Up to two billion times acceleration of scientific simulations with deep neural architecture search.
    arXiv:2001.08055v1

[2] Dimitrios Stamoulis, Ruizhou Ding, Di Wang,
    Dimitrios Lymberopoulos, Bodhi Priyantha,
    Jie Liu, Diana Marculescu
    Single-Path NAS: Device-Aware Efficient ConvNet Design.
    Arxiv:1905.04159v1
"""

# Imports
from __future__ import annotations
from typing import List, Any, Union
import torch
import torch.nn as nn
import numpy as np # type: ignore

# Helper functions
class Dict(dict):
    """ Helper class for avoiding using strings in the other classes"""
    def __getattr__(self, name: str) -> Any: return self[name]
    def __setattr__(self, name: str, value: Any) -> None: self[name] = value
    def __delattr__(self, name: str) -> None: del self[name]

def is_cuda() -> bool:
    """ Check if CuDa is available"""
    return torch.cuda.is_available()

def to_cuda(x: torch.Tensor) -> torch.Tensor:
    """ Converts tensor to cuda if cuda is available"""
    if is_cuda():
        return x.cuda()
    return x

def reshape_square(x: torch.Tensor) -> torch.Tensor:
    """ Returns n x n of x"""
    size = int(np.round(np.sqrt(x.shape[-1])))
    return to_cuda(
        torch.reshape(x, (-1, 1, size, size))
    )

def create_sub_blocks(in_channels: int, out_channels: int, default_param: dict) -> List[SuperConvBlock]:
    """ Create pairs of SuperConvBlocks"""
    return [
        SuperConvBlock(in_channels=in_channels, out_channels=out_channels, **default_param),
        SuperConvBlock(in_channels=out_channels, out_channels=out_channels, **default_param)
    ]

class SuperConv2d(nn.Module):
    """
    Class for implementing super convolution in 2d basesd on:
    'Single-Path NAS: Device-Aware Efficient ConvNet Design'
    (https://arxiv.org/abs/1905.04159v1)
    """
    def __init__(
        self, in_channels: int, out_channels=Union[int,None],
        out_channels_list: Union[list, None] = None, kernel_size=None,
        kernel_size_list: Union[list, None] = None, padding: int = 0,
        stride: int = 1, dilation: int = 1, groups:int = 1, bias: bool = True
    ) -> None:
        
        # Setup variables
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_list: list = [] if out_channels_list is None else out_channels_list
        self.kernel_size = kernel_size
        self.kernel_size_list: list = [] if kernel_size_list is None else kernel_size_list
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Compute maximum values
        max_out_channels = max(out_channels_list) if out_channels_list else self.out_channels
        max_kernel_size = max(kernel_size_list) if kernel_size_list else kernel_size

        # Setup channel masks
        channel_masks = []
        prev_out_channels = None
        for out_channels in self.out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            if prev_out_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels), [0, max_out_channels - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = out_channels
            channel_masks.append(channel_mask)

        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if out_channels_list else None)
        self.register_parameter('channel_thresholds', nn.Parameter(torch.ones(len(out_channels_list))) if out_channels_list else None)

        # Setup kernal masks
        kernel_masks = []
        prev_kernel_size = None
        for kernel_size in self.kernel_size_list:
            if kernel_size == 0:
                kernel_mask = torch.zeros(max_kernel_size, max_kernel_size)
            else:
                kernel_mask = torch.ones(max_kernel_size, max_kernel_size)
                kernel_mask *= nn.functional.pad(torch.ones(kernel_size, kernel_size), [(max_kernel_size - kernel_size) // 2] * 4, value=0)
            if prev_kernel_size:
                kernel_mask *= nn.functional.pad(torch.zeros(prev_kernel_size, prev_kernel_size), [(max_kernel_size - prev_kernel_size) // 2] * 4, value=1)
            kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(0)
            prev_kernel_size = kernel_size
            kernel_masks.append(kernel_mask)

        self.register_buffer('kernel_masks', torch.stack(kernel_masks, dim=0) if kernel_size_list else None)
        self.register_parameter('kernel_thresholds', nn.Parameter(torch.ones(len(kernel_size_list))) if kernel_size_list else None)
        self.register_parameter('weight', nn.Parameter(torch.Tensor(max_out_channels, self.in_channels // groups, max_kernel_size, max_kernel_size)))
        self.register_parameter('bias', nn.Parameter(torch.Tensor(max_out_channels)) if bias else None)

        self.max_out_channels = max_out_channels
        self.max_kernel_size = max_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass"""
        weight = self.weight
        if self.channel_masks is not None and self.channel_thresholds is not None:
            weight = weight * self.parametrized_mask(list(self.channel_masks), list(self.channel_thresholds)) # type: ignore
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            weight = weight * self.parametrized_mask(list(self.kernel_masks), list(self.kernel_thresholds)) # type: ignore
        return nn.functional.conv2d(
            x, weight,                          # type: ignore
            self.bias, padding=self.padding,    # type: ignore
            stride=self.stride, dilation=self.dilation,
            groups=self.groups
        )
    def freeze_weight(self) -> torch.Tensor:
        weight = self.weight.detach().clone()
        if self.channel_masks is not None and self.channel_thresholds is not None:
            prev_out_channels = None
            for channel_mask, channel_threshold, out_channels in zip(self.channel_masks, self.channel_thresholds, self.out_channels_list):
                if prev_out_channels:
                    channel_norm = torch.norm(self.weight * channel_mask)
                    if channel_norm < channel_threshold:
                        weight = weight[..., :prev_out_channels]
                        break
                prev_out_channels = out_channels
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            prev_kernel_size = None
            for kernel_mask, kernel_threshold, kernel_size in zip(self.kernel_masks, self.kernel_thresholds, self.kernel_size_list):
                if prev_kernel_size:
                    kernel_norm = torch.norm(self.weight * kernel_mask)
                    if kernel_norm < kernel_threshold:
                        cut = (self.max_kernel_size - prev_kernel_size) // 2
                        weight = weight[..., cut:-cut, cut:-cut]
                        break
                prev_kernel_size = kernel_size
        return weight

    def parametrized_mask(self, masks: List[torch.Tensor], thresholds: List[torch.Tensor]) -> Union[torch.Tensor, int]:
        """ Parametrizes the masks for the 2d convolution."""
        if not masks or not thresholds:
            return 0
        mask = masks.pop(0)
        threshold = thresholds.pop(0)
        norm = torch.norm(self.weight * mask)
        indicator = (norm > threshold).float() - torch.sigmoid(norm - threshold).detach() + torch.sigmoid(norm - threshold)
        return indicator * (mask + self.parametrized_mask(masks, thresholds))

    def get_kernel(self) -> torch.Tensor:
        """ Returns the kernel mask used."""
        return self.parametrized_mask(list(self.kernel_masks), list(self.kernel_thresholds))
    
    @staticmethod
    def get_link() -> str:
        """ Return the link to the paper"""
        return "https://arxiv.org/abs/1905.04159v1"
    @staticmethod
    def get_title() -> str:
        """ Return the title of the paper"""
        return "Single-Path NAS: Device-Aware Efficient ConvNet Design"

class SuperConvBlock(nn.Module):
    """
    Class for creating SuperConv2d blocks (Multiple conv layers inside each block)
    based on MBConv mirco-arc on:
    'Single-Path NAS: Device-Aware Efficient ConvNet Design'
    (https://arxiv.org/abs/1905.04159v1)
    """
    def __init__(self, in_channels: int, out_channels: int, expand_ratio_list: list, kernel_size_list: list, stride: int) -> None:

        # Init and compute internal values
        super().__init__()
        hidden_channels_list = [in_channels * expand_ratio for expand_ratio in expand_ratio_list]
        max_hidden_channels = max(hidden_channels_list)
        max_kernel_size = max(kernel_size_list)
        
        # Setup the SuperConvBlock
        self.module = nn.Sequential(
            # First layer
            nn.Sequential(
                SuperConv2d(
                    in_channels=in_channels,
                    out_channels_list=hidden_channels_list,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            # Second layer
            nn.Sequential(
                SuperConv2d(
                    in_channels=max_hidden_channels,
                    out_channels=max_hidden_channels,
                    groups=max_hidden_channels,
                    kernel_size_list=kernel_size_list,
                    padding=(max_kernel_size - 1) // 2,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            # Third layer
            nn.Sequential(
                nn.Conv2d(
                    in_channels=max_hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        )
        # Create a shortcut and upcase of condition is meet
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes the forward pass"""
        output = self.module(x)
        # Add residual
        output += self.shortcut(x)
        return output
    @staticmethod
    def get_link() -> str:
        """ Return the link to the paper"""
        return "https://arxiv.org/abs/1905.04159v1"
    @staticmethod
    def get_title() -> str:
        """ Return the title of the paper"""
        return "Single-Path NAS: Device-Aware Efficient ConvNet Design"

class SuperArchitectureNoLinear(nn.Module):
    """
    Creates the SuperArchitecture from the following paper without the initial linear layers
    'Up to two billion times acceleration of scientific simulations with deep neural architecture search'
    (https://arxiv.org/abs/2001.08055v1)
    """
    def __init__(
        self,
        middle_conv_params: List[Dict],
        last_conv_param: List[Dict],
        last_linear_params: List[Dict]
    ) -> None:
        # Init
        super().__init__()

        # Middle blocks
        self.middle = nn.Sequential(
                *[
                    SuperConvBlock(
                        in_channels=middle_conv_param.in_channels,
                        out_channels=middle_conv_param.out_channels,
                        expand_ratio_list=middle_conv_param.expand_ratio_list,
                        kernel_size_list=middle_conv_param.kernel_size_list,
                        stride=middle_conv_param.stride
                    ) for middle_conv_param in middle_conv_params
                ]
        )

        # output convolution
        self.end = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_conv_param.in_channels,            # type: ignore
                    out_channels=last_conv_param.out_channels,          # type: ignore
                    kernel_size=last_conv_param.kernel_size,            # type: ignore
                    padding=(last_conv_param.kernel_size - 1) // 2,     # type: ignore
                    stride=last_conv_param.stride,                      # type: ignore
                    bias=False
                ),
                nn.BatchNorm2d(last_conv_param.out_channels),           # type: ignore
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
            )
        )
        
        # Output layer
        self.out = nn.Sequential(
            *[nn.Linear(**params) for params in last_linear_params]
        )
        
        # Initialize layers
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, SuperConv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)                         # type: ignore
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the forward pass"""
        x_middle = self.middle(x)
        x_end = self.end(x_middle)
        x_flatten_out = x_end.view(x_end.size(0), -1)
        x_out = self.out(x_flatten_out)
        return x_out

    def weights(self):
        """ Returns weights of the model"""
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter

    def thresholds(self):
        """ Returns the thresholds of the model""" 
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter
    @staticmethod
    def get_link() -> str:
        """ Return the link to the paper"""
        return "https://arxiv.org/abs/2001.08055v1"
    @staticmethod
    def get_title() -> str:
        """ Return the title of the paper"""
        return "Up to two billion times acceleration of scientific simulations with deep neural architecture search"
    
class SuperArchitecture(nn.Module):
    """
    Creates the SuperArchitecture from the following paper
    'Up to two billion times acceleration of scientific simulations with deep neural architecture search'
    ("https://arxiv.org/abs/2001.08055v1")
    """
    def __init__(
        self,
        first_linear_params: List[Dict],
        middle_conv_params: List[Dict],
        last_conv_param: List[Dict],
        last_linear_params: List[Dict]
    ) -> None:
        # Init
        super().__init__()
        
        # Linear initial layers
        self.linear = nn.Sequential(
            *[
                nn.Linear(
                    in_features=params.in_features,
                    out_features=params.out_features
                ) for params in first_linear_params
            ],
            nn.BatchNorm1d(first_linear_params[-1].out_features),
            nn.ReLU6()
        )
        
        # Middle blocks
        self.middle = nn.Sequential(
                *[
                    SuperConvBlock(
                        in_channels=middle_conv_param.in_channels,
                        out_channels=middle_conv_param.out_channels,
                        expand_ratio_list=middle_conv_param.expand_ratio_list,
                        kernel_size_list=middle_conv_param.kernel_size_list,
                        stride=middle_conv_param.stride
                    ) for middle_conv_param in middle_conv_params
                ]
        )

        # output convolution
        self.end = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_conv_param.in_channels,            # type: ignore
                    out_channels=last_conv_param.out_channels,          # type: ignore
                    kernel_size=last_conv_param.kernel_size,            # type: ignore
                    padding=(last_conv_param.kernel_size - 1) // 2,     # type: ignore
                    stride=last_conv_param.stride,                      # type: ignore
                    bias=False
                ),
                nn.BatchNorm2d(last_conv_param.out_channels),           # type: ignore
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
            )
        )
        
        # Output layer
        self.out = nn.Sequential(
            *[nn.Linear(**params) for params in last_linear_params]
        )
        
        # Initialize layers
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, SuperConv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)                         # type: ignore
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the forward pass"""
        x_flatten_in = x.view(x.size(0), -1)
        x_lin = self.linear(x_flatten_in)
        x_square = reshape_square(x_lin)
        x_middle = self.middle(x_square)
        x_end = self.end(x_middle)
        x_flatten_out = x_end.view(x_end.size(0), -1)
        x_out = self.out(x_flatten_out)
        return x_out
    
    def weights(self):
        """ Returns weights of the model"""
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter

    def thresholds(self):
        """ Returns the thresholds of the model""" 
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter
    
    @staticmethod
    def get_link() -> str:
        """ Return the link to the paper"""
        return "https://arxiv.org/abs/2001.08055v1"
    @staticmethod
    def get_title() -> str:
        """ Return the title of the paper"""
        return "Up to two billion times acceleration of scientific simulations with deep neural architecture search"