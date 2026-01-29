import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Standard embedding layer for mapping input tokens to a continuous vector space.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the embedding layer.

        Args:
            d_model: Dimensionality of each embedding vector.
            vocab_size: Total number of unique tokens in the vocabulary.
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # PyTorch provides an embedding layer as a lookup table.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the embeddings for the input tensor and applies scaling.
        """
        # NOTE: Scale the embeddings by the square root of d_model to stabilize gradients.
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative and absolute position of tokens in the sequence.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (seq_len, d_model) to store positional encodings
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1) representing positions (0, 1, 2, ..., seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the frequency scaling factor in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices (0, 2, 4, ...) and cosine to odd indices (1, 3, 5, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer so it's saved with the model but not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input embeddings and applies dropout.
        """
        # Add the precomputed encoding to the input (sliced to match current sequence length)
        # We use .detach() to ensure these values are treated as constants
        x = x + self.pe[:, :x.shape[1], :].detach()
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over the last dimension of the input tensor.
    This helps stabilize the training of deep neural networks by normalizing activations.
    """

    def __init__(self, eps: float = 10**-6) -> None:
        """
        Args:
            eps: A small value added to the denominator for numerical stability to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        # Alpha is a learnable scaling parameter (gamma), initialized to 1.
        self.alpha = nn.Parameter(torch.ones(1))
        # Bias is a learnable shift parameter (beta), initialized to 0.
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor along the feature dimension.
        """
        # Calculate mean and standard deviation across the last dimension (d_model).
        # keepdim=True allows for correct broadcasting during subtraction and division.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Standardize the input and apply the learnable scale (alpha) and shift (bias).
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) as described in the Transformer architecture.
    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: Dimensionality of the input and output embeddings.
            d_ff: Dimensionality of the internal hidden layer (e.g., 2048).
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
    
    def forward(self, x):
        return self.ff(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention mechanism as described in 'Attention Is All You Need'.
    Allows the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Ensure the embedding dimension is divisible by the number of heads for equal splitting
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of each individual attention head

        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # NOTE [wrong]: Each Self Attention block maintains its own independent set of weight matrices.

        # Output projection layer to combine head outputs
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout: nn.Dropout):
        """
        Computes Scaled Dot-Product Attention.
        """
        d_k = query.shape[-1]

        # Calculate raw attention scores: (batch, h, seq_len, seq_len)
        # Scaling by sqrt(d_k) prevents large values from saturating the softmax
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # NOTE: Need to check the shapes

        # Apply mask (padding or look-ahead) by setting masked positions to a very small value
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Convert scores to probabilities along the sequence dimension
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the weighted sum of values: (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 1. Project inputs into Q, K, V spaces
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)

        # 2. Split into 'h' heads for parallel attention
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # 3. Apply Scaled Dot-Product Attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # 4. Concatenate heads back together
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        # .contiguous() is needed before .view() because transpose() modifies the memory layout
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 5. Final linear projection back to d_model
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """
    Implements a residual connection around a sublayer (Attention or FeedForward).
    This follows the 'Pre-LayerNorm' design where normalization is applied before the sublayer.
    """
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            sublayer: A callable (module or lambda) that processes the normalized input
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model) after residual addition
        """
        # 1. self.norm(x): (batch, seq_len, d_model) -> Normalizes the input
        # 2. sublayer(...): (batch, seq_len, d_model) -> Applies Attention or FeedForward
        # 3. self.dropout(...): (batch, seq_len, d_model) -> Applies dropout to sublayer result
        # 4. x + ...: (batch, seq_len, d_model) -> Adds the original input (Residual connection)
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    A single block of the Encoder, containing a Multi-Head Self-Attention layer
    and a Position-wise Feed-Forward Network, each followed by a residual connection.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Create two residual connections: 
        # Index 0 handles the Self-Attention sublayer, Index 1 handles the Feed-Forward sublayer.
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Mask to ignore padding tokens, typically shape (batch, 1, 1, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Sublayer 1: Multi-Head Self-Attention
        # We use a lambda to pass query, key, value (all 'x') and the mask to the attention block.
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        """
            a = lambda x: x + 5
            print(a(10)) -> 15
        """

        # Sublayer 2: Feed-Forward Network
        # The residual connection applies LayerNorm, then the FFN, then dropout and addition.
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x

class Encoder(nn.Module):
    """
    The Encoder consists of a stack of N EncoderBlocks.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        # Store the stack of encoder layers (EncoderBlocks).
        self.layers = layers
        # Final layer normalization applied after the stack of blocks.
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Source mask to ignore padding tokens
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pass the input through each encoder block in the stack sequentially.
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply the final layer normalization to the output of the last block.
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    A single block of the Decoder, containing Masked Multi-Head Self-Attention,
    Multi-Head Cross-Attention (with Encoder output), and a Position-wise Feed-Forward Network.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
            x: decoder input (query)
            encoder_output: (key, value)
            src_mask: encoder mask
            tgt_mask: decoder mask
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):
    """
    The Decoder consists of a stack of N DecoderBlocks.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        # Store the stack of decoder layers (DecoderBlocks).
        self.layers = layers
        # Final layer normalization applied after the stack of blocks.
        self.norm = LayerNormalization()
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            encoder_output: Output from the encoder stack
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pass the input through each decoder block in the stack sequentially.
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply the final layer normalization to the output of the last block.
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projects the model's output dimension (d_model) to the vocabulary size.
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # Linear layer to map d_model to vocab_size.
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear projection followed by a log-softmax.
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)
    # NOTE: We dont need two positional encoders since the values belongs to each positon is same

    # Create the encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, vocab_size=tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initializing the weights with xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

     






def main() -> None:
    # Input
    sentence = torch.tensor([34, 98, 64, 55, 19])

    # Input Embeddings
    word_embedding = InputEmbeddings(d_model=512, vocab_size=100)
    x = word_embedding(sentence)
    print("Input Embeddings \n", x, end="\n\n") # shape -> (5, 512)

    # Positional Encodings
    positional_embedding = PositionalEncoding(d_model=512, seq_len=5, dropout=0.2)
    x = positional_embedding(x)
    print("Positional Encodings \n", x, end="\n\n") # shape -> (1, 5, 512) - Added batch dimention

    # Multi-Head Attention
    multi_head_attention = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.2)
    """
        x = multi_head_attention(x, x, x, None)
        print("Multi-Head Attention \n", x)
    """

    # Feed Forward Neural Network
    feed_forward_nn = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.2)
    
    # Encoder Block
    encoder_block = EncoderBlock(
        self_attention_block=multi_head_attention,
        feed_forward_block=feed_forward_nn,
        dropout=0.2
    )
    print(encoder_block(x, None), end="\n\n")


if __name__ == "__main__":
    main()