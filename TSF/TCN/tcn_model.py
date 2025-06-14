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
            input_dim    (int)      : Number of input features per time step
            hidden_dims  (List[int]): List of TCN layer widths
            output_steps (int)      : Number of time steps to forecast
            kernel_size  (int)      : Convolution kernel size
            dropout      (float)    : Dropout rate in TCN
            num_heads    (int)      : Number of attention heads
        """

    def __init__(
        self,
        input_dim    : int,
        hidden_dims  : List[int],
        output_steps : int,
        kernel_size  : int = 3,
        dropout      : float = 0.2,
        num_heads    : int = 2
    ):

        super().__init__()

        self.encoder = TCNEncoder(
            in_channels  = input_dim,
            num_channels = hidden_dims,
            kernel_size  = kernel_size,
            dropout      = dropout
        )
        self.decoder = AttentionDecoder(
            hidden_dim   = hidden_dims[-1],
            output_steps = output_steps,
            num_series   = input_dim,
            num_heads    = num_heads
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
