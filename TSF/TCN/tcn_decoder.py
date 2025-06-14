import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    
    """
        Attention-based Decoder for sequence-to-sequence forecasting.
            - Multi-head self-attention to compute context-aware representations.
            - Linear projection to map attention outputs to forecasts.    

        Args:
            hidden_channels (int) : Dimension of encoder outputs and attention embedding.
            output_channels (int) : Number of output dimensions
            output_len      (int) : Number of future time steps to forecast.
            num_heads       (int) : Number of attention heads.
    """

    def __init__(
        self,
        hidden_channels : int,
        output_channels : int, 
        output_len      : int,
        num_heads       : int = 2
    ):

        super().__init__()

        self.output_len = output_len

        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_channels,
            num_heads   = num_heads,
            batch_first = True 
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)  # Predict ALL values per time step
        )

    def forward(
        self, 
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        
        """
            Args:
                encoder_outputs: (batch_size, seq_len, hidden_channels)
        
            Returns:
                    forecast: (batch_size, output_len, output_channels)
        """
        
        batch_size, seq_len, hidden_channels = encoder_outputs.shape

        query = torch.zeros(
                    batch_size, 
                    self.output_len, 
                    hidden_channels,
                    device = encoder_outputs.device
                )

        attn_output, _ = self.attn(query, encoder_outputs, encoder_outputs)

        return self.fc(attn_output)  # (batch_size, output_len, output_channels)

