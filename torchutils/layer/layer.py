import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchsummary import summary

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_att)
        x_att = self.bn(x_att)
        return self.sigmoid(x_att) * x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel // reduction)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg = self.relu(self.bn1(self.conv1(avg)))
        max = self.relu(self.bn1(self.conv1(max)))
        avg = self.bn2(self.conv2(avg))
        max = self.bn2(self.conv2(max))
        return self.sigmoid(avg + max) * x


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Ajout de la connexion résiduelle
        out = self.relu(out)
        return out


def make_layer(block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)


class LearnableRandomNoise(nn.Module):
    def __init__(self, channels):
        super(LearnableRandomNoise, self).__init__()
        self.coef = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        coef_scale = torch.clamp(self.coef, min=0, max=1).view(1, -1, 1, 1)
        return x + (torch.randn_like(x, device=x.device, dtype=x.dtype, min_val=0, max_val=255) * coef_scale)


class LearnDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Liste des groupes possibles (diviseurs communs)
        self.divisors = self.common_divisors(in_channels, out_channels)
        self.num_options = len(self.divisors)

        # Logits pour pondérer les groupes
        self.group_logits = nn.Parameter(torch.randn(self.num_options))

        # Un poids par option de groupe
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, in_channels // g, *self.kernel_size))
            for g in self.divisors
        ])

        # Bias partagé (optionnel : un par config si besoin)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        probs = F.softmax(self.group_logits, dim=0)  # [num_groups]

        outputs = []
        for p, w, g in zip(probs, self.weights, self.divisors):
            out = F.conv2d(x, w, self.bias, groups=g, stride=self.stride, padding=self.padding)
            outputs.append(p * out)

        return sum(outputs)

    def common_divisors(self, a, b):
        return [d for d in range(1, min(a, b) + 1) if a % d == 0 and b % d == 0]


class EmbedPatchImage(nn.Module):
    def __init__(self, patch_size, embed_size):
        super(EmbedPatchImage, self).__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.fc = nn.Linear(patch_size, embed_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        num_patches = x.size(2) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)
        x = self.fc(x)
        return x