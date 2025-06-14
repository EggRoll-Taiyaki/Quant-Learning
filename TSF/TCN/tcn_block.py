import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class TCNBlock(nn.Module):

    """
        Temporal Convolutional Block (TCN Block):
            - Causal 1D convolutional layers with exponentially increasing dilation.
            - Weight normalization for faster convergence.
            - ReLU activation for non-linearity.
            - Dropout for regularization.
            - Residual connection to stabilize gradient flow.

        Args:
            in_channels  (int): Number of input channels (features per time step).
            out_channels (int): Number of output channels after convolution.
            kernel_size  (int): Size of the convolutional filter.
            dilation     (int): Dilation rate for expanding receptive field.
            dropout    (float): Dropout rate between conv layers.
    """


    def __init__(
        self, 
        in_channels  : int, 
        out_channels : int, 
        kernel_size  : int, 
        dilation     : int,
        dropout      : float = 0.2
    ):
        
        super().__init__()
        
        self.conv = weight_norm(nn.Conv1d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size - 1) * dilation,
            dilation     = dilation
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # apply a 1x1 convolution to match dimensions for residual addition
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size = 1)
                if in_channels != out_channels else None
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:

        """
            Args:
                x: Input tensor of shape (batch_size, in_channels, seq_len)

            Returns:
                 Output tensor of shape (batch_size, out_channels, seq_len)
        """

        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]] # Remove future-leaking padding
        out = self.relu(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        return out + residual
