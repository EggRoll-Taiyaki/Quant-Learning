import torch
import torch.nn as nn
from typing import List
from tcn_encoder import TCNEncoder
from tcn_decoder import AttentionDecoder

class TCNModel(nn.Module):
    
    """
        Time Series Forecasting Model using:
            - TCN Encoder to extract temporal features from historical data
            - Attention Decoder to predict future values

        Args:
            in_channels     (int)      : Number of input features per time step
            hidden_channels (List[int]): List of TCN layer widths
            output_len      (int)      : Number of time steps to forecast
            kernel_size     (int)      : Convolution kernel size
            dropout         (float)    : Dropout rate in TCN
            num_heads       (int)      : Number of attention heads
        """

    def __init__(
        self,
        in_channels     : int,
        hidden_channels : List[int],
        output_len      : int,
        kernel_size     : int = 3,
        dropout         : float = 0.2,
        num_heads       : int = 2
    ):

        super().__init__()

        self.encoder = TCNEncoder(
            in_channels     = in_channels,
            hidden_channels = hidden_channels,
            kernel_size     = kernel_size,
            dropout         = dropout
        )
        self.decoder = AttentionDecoder(
            hidden_channels = hidden_channels[-1],
            output_len      = output_len,
            out_channels      = in_channels,
            num_heads       = num_heads
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        """
            Args:
                x: (batch_size, seq_len, feature_dim)
        
            Returns:
                 forecast: (batch_size, output_steps)
        """
        
        return self.decoder(self.encoder(x))
