import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Tuple, List, Optional

from basicsr.utils.registry import ARCH_REGISTRY

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion: int = 1  # output channel expansion ratio

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__() # Python 3 super()
        self.conv1: nn.Conv2d = conv3x3(in_planes, planes, stride)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = conv3x3(planes, planes)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = downsample
        self.stride: int = stride

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):

    expansion: int = 1  # output channel expansion ratio

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, use_se: bool = True):
        super().__init__() # Python 3 super()
        self.bn0: nn.BatchNorm2d = nn.BatchNorm2d(in_planes)
        self.conv1: nn.Conv2d = conv3x3(in_planes, in_planes) # Note: original was in_planes, in_planes - seems correct for IR block pre-activation
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(in_planes) # Matched to conv1 output
        self.prelu: nn.PReLU = nn.PReLU()
        self.conv2: nn.Conv2d = conv3x3(in_planes, planes, stride) # Output channels should be 'planes'
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = downsample
        self.stride: int = stride
        self.use_se: bool = use_se
        if self.use_se:
            self.se: SEBlock = SEBlock(planes) # SEBlock operates on 'planes' channels

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = x
        
        out: Tensor = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out) # Second PReLU after residual add

        return out


class Bottleneck(nn.Module):

    expansion: int = 4  # output channel expansion ratio

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__() # Python 3 super()
        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.conv3: nn.Conv2d = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(planes * self.expansion)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = downsample
        self.stride: int = stride

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__() # Python 3 super()
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)  # pool to 1x1
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y: Tensor = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


@ARCH_REGISTRY.register()
class ResNetArcFace(nn.Module):

    def __init__(self, block_name: str, layers: Tuple[int, int, int, int], use_se: bool = True):
        super().__init__() # Python 3 super()

        self.block: Type[BasicBlock | IRBlock | Bottleneck] # More specific type for the block class
        if block_name == 'IRBlock':
            self.block = IRBlock
        elif block_name == 'BasicBlock': # Added for completeness
            self.block = BasicBlock
        elif block_name == 'Bottleneck': # Added for completeness
            self.block = Bottleneck
        else:
            raise ValueError(f"Unknown block_name: {block_name}. Choose from 'IRBlock', 'BasicBlock', 'Bottleneck'.")

        self.in_planes: int = 64 # Renamed from inplanes for consistency
        self.use_se: bool = use_se

        # Initial convolution layers
        self.conv1: nn.Conv2d = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False) # Assuming 1-channel (grayscale) input
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.prelu: nn.PReLU = nn.PReLU()
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # ResNet layers
        self.layer1: nn.Sequential = self._make_layer(self.block, 64, layers[0])
        self.layer2: nn.Sequential = self._make_layer(self.block, 128, layers[1], stride=2)
        self.layer3: nn.Sequential = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4: nn.Sequential = self._make_layer(self.block, 512, layers[3], stride=2)

        # Final layers
        self.bn4: nn.BatchNorm2d = nn.BatchNorm2d(512 * self.block.expansion) # Adjusted for block expansion
        self.dropout: nn.Dropout = nn.Dropout()
        # The input feature size to fc5 depends on the output of layer4 and maxpool.
        # Assuming input image size and network structure leads to 8x8 feature maps here.
        # E.g., for 128x128 input: 128 -> 64 (maxpool) -> 64 (layer1) -> 32 (layer2) -> 16 (layer3) -> 8 (layer4)
        # The 512 comes from 512 * block.expansion.
        self.fc5: nn.Linear = nn.Linear(512 * self.block.expansion * 8 * 8, 512)
        self.bn5: nn.BatchNorm1d = nn.BatchNorm1d(512)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)): # Grouped types
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block_class: Type[BasicBlock | IRBlock | Bottleneck], # Use the specific block class type
        planes: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.in_planes != planes * block_class.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block_class.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_class.expansion),
            )

        layer_list: List[nn.Module] = [] # Use generic list for type hint
        
        # For IRBlock, 'use_se' is a parameter. For others, it's not directly used in the constructor signature
        # shown here, but the main class's self.use_se controls it if IRBlock is chosen.
        block_kwargs = {'in_planes': self.in_planes, 'planes': planes, 'stride': stride, 'downsample': downsample}
        if block_class == IRBlock: # Only pass use_se if it's an IRBlock
            block_kwargs['use_se'] = self.use_se
        
        layer_list.append(block_class(**block_kwargs))
        
        self.in_planes = planes * block_class.expansion # Update in_planes for the next block in this layer or next layer
        
        for _ in range(1, num_blocks):
            block_kwargs_iter = {'in_planes': self.in_planes, 'planes': planes} # stride=1, downsample=None implicitly
            if block_class == IRBlock:
                 block_kwargs_iter['use_se'] = self.use_se
            layer_list.append(block_class(**block_kwargs_iter))
            # self.in_planes remains planes * block_class.expansion for subsequent blocks in the same layer

        return nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc5(x)
        x = self.bn5(x)

        return x