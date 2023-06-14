import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class LeakySineLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(0.1 * (torch.sin(x) ** 2 + x), torch.sin(x) ** 2 + x)


class ResidualModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, transpose: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convolution_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_1 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_2 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_3 = nn.BatchNorm1d(num_features=out_channels)
        
        self.activation = LeakySineLU()
        
        self.shortcut = nn.Sequential(*[
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(num_features=out_channels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.activation(self.batchnorm_1(self.convolution_1(x)))
        x_ = self.activation(self.batchnorm_2(self.convolution_2(x_)))
        x_ = self.batchnorm_3(self.convolution_3(x_))

        return self.activation(x_ + self.shortcut(x))
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modules: int) -> None:
        super().__init__()

        self.residual_modules = nn.ModuleList()
        self.residual_modules.append(
            ResidualModule(in_channels=in_channels, out_channels=out_channels)
        )
        c_in, c_out = out_channels, out_channels * 2
        for i in range(num_modules - 1):
            self.residual_modules.append(
                ResidualModule(in_channels=c_in, out_channels=c_out)
            )
            c_in = c_out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.residual_modules:
            x = module(x)
        return x


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, representation_dim: int) -> None:
        super().__init__()
        self.residual_net = ResidualBlock(in_channels=in_channels, out_channels=out_channels, num_modules=3)
        self.fc = nn.Linear(in_features=out_channels, out_features=representation_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(x) # ResNetEncoder
        x = torch.mean(x, dim=0) # GAP
        return self.fc(x)


class ResidualTransposeModule(nn)
