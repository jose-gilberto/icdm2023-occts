from typing import Any, Tuple
from torch import nn
import torch
import pytorch_lightning as pl


class LSTMAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        project_channels: int,
        hidden_channels: int,
        window_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.project_channels = project_channels
        self.hidden_channels = hidden_channels
        self.window_size = window_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = nn.LSTM(
            self.in_channels, self.hidden_channels, batch_first=True,
            num_layers=self.num_layers, bias=False, dropout=self.dropout
        )
        
        self.decoder = nn.LSTM(
            self.in_channels, self.hidden_channels, batch_first=True,
            num_layers=self.num_layers, bias=False, dropout=self.dropout
        )
        
        self.output_layer = nn.Linear(self.hidden_channels, self.in_channels)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2, bias=False),
            nn.BatchNorm1d(self.hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels // 2, self.project_channels, bias=False)
        )
        
    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...