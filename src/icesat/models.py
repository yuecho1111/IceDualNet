"""
Neural network models for ICESat classification.

Implements the DualPathNetwork architecture with local and context paths.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DualPathNetwork(nn.Module):
    """Dual-path neural network for sea ice classification.
    
    Combines:
    - Local path: Physical features of the center point
    - Context path: Waveform context from entire sequence
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, 
                 sequence_length: int = 7, dropout_rate: float = 0.3):
        """Initialize the network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            sequence_length: Length of input sequences
            dropout_rate: Dropout probability
        """
        super(DualPathNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Local path: Process center point features
        self.local_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Context path: Extract waveform trends
        self.context_path = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion: Combine local and context features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output logits of shape (batch_size,)
        """
        batch, seq, feat = x.shape
        mid_idx = seq // 2

        # Local path: Process center point
        local_feat = self.local_path(x[:, mid_idx, :])  # (batch, hidden_dim)

        # Context path: Process entire sequence
        context_in = x.permute(0, 2, 1)  # (batch, input_dim, seq)
        context_feat = self.context_path(context_in).squeeze(-1)  # (batch, hidden_dim)

        # Fusion: Concatenate and process
        fused = torch.cat([local_feat, context_feat], dim=-1)
        return self.fusion(fused).squeeze(-1)  # (batch,)
