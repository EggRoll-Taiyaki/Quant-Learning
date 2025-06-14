import torch
import torch.nn as nn
from tcn_block import TCNBlock

class TCNEncoder(nn.Module):
    
    """
        Stacked TCN layers as a sequence encoder.
        Expands receptive field using increasing dilations in each layer.
        
        Args:
            in_channels     (int)       : Number of input features
            hidden_channels (list[int]) : List of channel sizes for each TCN layer
            kernel_size     (int)       : Size of the temporal convolution kernel
            dropout         (float)     : Dropout rate between conv layers
    """

    def __init__(
        self, 
        in_channels     : int, 
        hidden_channels : list[int], 
        kernel_size     : int = 3,
        dropout         : float = 0.2
    ):
        
        super().__init__()
        
        layers = []
        for i in range(len(hidden_channels)):
            in_ch   = in_channels if i == 0 else hidden_channels[i - 1]
            out_ch  = hidden_channels[i]
            layers += [ 
                TCNBlock(
                    in_channels  = in_ch, 
                    out_channels = out_ch, 
                    kernel_size  = kernel_size, 
                    dilation     = 2**i
                ) 
            ] 

        self.network = nn.Sequential(*layers)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        """
            Args:
                x: Input tensor of shape (batch_size, seq_len, in_dim)

            Returns:
                 Output tensor of shape (batch_size, seq_len, out_dim)
        
            Although PyTorch Conv1d expects shape (batch, channels, seq_len),
            sequence models prefer shape (batch, seq_len, channels) instead.
        """

        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.permute(0, 2, 1)
        return out
