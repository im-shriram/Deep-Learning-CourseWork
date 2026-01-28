import math
from turtle import forward
from typing import Any
from sympy import ff
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
            d_model: dimentionality of each embedding vector
            vocab_size: Number of unique words in your dataset
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) # PyTorch already provide the embedding layer - lookup table
    
    def forward(self, x) -> Any:
        return self.embedding(x) * math.sqrt(self.d_model)

class PosisionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout) -> None:
        """
            d_model: dimentionality of positional encodings
            seq_len: max document length
            dropout: NOTE -> How it works?
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create a matrix of shape (seq_len, dropout)
        pe = torch.zeros(seq_len, d_model) # Each row represents the positional encoding of each position
        # Create a numerator and denomator of the formula
        position = torch.arange(start=0, end=seq_len, step=1, dtype=torch.float).unsqueeze(dim=1) # shape -> (seq_len , 1)
        div_term = torch.exp(
            torch.arange(start=0, end=d_model, step=2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        ) # This formula is slightly different than the original formula -> This is numerically more stable than the original formula.

        # Apply the sin and cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # We are processing every document in batches -> adding batch dimention
        pe = pe.unsqueeze(dim=0) # shape -> (1, seq_len, d_model)

        # NOTE: What is this?
        self.register_buffer('pe', pe)
        self.pe = pe
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Basically, we are using only positional embedding of number of words present in the current document.
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.tensor(1)) # Added
        # nn.Parameter() makes tensor learnable
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Last 2nd dimention
        std = x.std(dim=-1, keepdim=True) 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
            d_model: dimentionality of embeddings
            d_ff: no. neurons in 1st hieedn layer → 2048
        """
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
    
    def forward(self, x): # The shape of x is in batches -> (batch, seq_len, d_model)
        return ff(x) # preserves the batch dim