import torch
import torch.nn as nn
import math
from torchinfo import summary

class InputEmbeddings(nn.Module):
    '''
        Embedding before going to the model
    '''
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model #dimension of the model
        self.vocab_size = vocab_size #Size of the dictionary for vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    '''
        Add the positional information to the input
    '''
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

# class LayerNormalization(nn.Module):
#     '''
#     Normalize the last dimension dynamically without requiring `features` at initialization.
#     '''
#     def __init__(self, d_model: int = 512, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(d_model))  # Khởi tạo đúng kích thước từ đầu
#         self.bias = nn.Parameter(torch.zeros(d_model))  # Khởi tạo đúng kích thước từ đầu

#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True)
#         return self.alpha * (x - mean) / (std + self.eps) + self.bias

class LayerNormalization(nn.Module):
    '''
        Normalize the last dimension dynamically without requiring `features` at initialization.
    '''
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = None
        self.bias = None

    def forward(self, x):
        # Dynamically initialize weight and bias based on the input shape
        if self.alpha is None or self.bias is None:
            features = x.size(-1)  # Last dimension
            self.alpha = nn.Parameter(torch.ones(features, device=x.device))
            self.bias = nn.Parameter(torch.zeros(features, device=x.device))

        # Calculate mean and std along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.head = head # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % head == 0, "d_model is not divisible by h"

        self.d_k = d_model // head # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9) #cái mask bằng bao nhiêu sẽ được định nghĩa ở sau
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head * self.d_k) #Contiguous tensors allow for faster computation since their elements are stored sequentially in memory.


        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
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
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 1. Encode source sequence
        encoder_output = self.encode(src, src_mask)  # (batch, src_seq_len, d_model)
        
        # 2. Decode target sequence
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)  # (batch, tgt_seq_len, d_model)
        
        # 3. Project decoder output to vocab space
        logits = self.project(decoder_output)  # (batch, tgt_seq_len, vocab_size)
        
        return logits
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, head: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.01)
        elif isinstance(module, LayerNormalization):
            if module.alpha is not None:
                nn.init.ones_(module.alpha)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Apply initialization
    transformer.apply(init_weights)
    
    return transformer

if __name__ == "__main__":
    #import torch

    # Example parameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    src_seq_len = 64
    tgt_seq_len = 64
    batch_size = 2

    # Create dummy data for source and target
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # Random source data
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # Random target data

    # Function to create padding mask
    def create_padding_mask(seq, pad_token=0):
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)

    # Function to create causal mask
    def create_causal_mask(size):
        # Create a lower triangular matrix
        return torch.tril(torch.ones((1, 1, size, size))).bool()

    # Create masks
    src_mask = create_padding_mask(src)  # Padding mask for source
    print(src_mask)
    tgt_pad_mask = create_padding_mask(tgt)  # Padding mask for target
    tgt_causal_mask = create_causal_mask(tgt_seq_len)  # Causal mask for target
    tgt_mask = tgt_pad_mask & tgt_causal_mask  # Combine padding and causal masks
    print(tgt_mask)
    # Initialize the model
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=512,
        N=6,
        head=8,
        d_ff=2048
    )

    # Forward pass with dummy data
    logits = model(src, tgt, src_mask, tgt_mask)
    print("Logits shape:", logits.shape)  # Expected shape: (batch_size, tgt_seq_len, tgt_vocab_size)

    # Inference function
    def translate_sentence(model, src, src_vocab_size, tgt_vocab_size, max_len=64, pad_token=0, start_token=1, end_token=2):
        model.eval()
        src_mask = create_padding_mask(src, pad_token=pad_token)  # Padding mask for source
        encoder_output = model.encode(src, src_mask)  # Encode source sequence

        tgt = torch.tensor([[start_token]], dtype=torch.long).to(src.device)  # Start with <START> token
        for _ in range(max_len):
            # Create masks for current target sequence
            tgt_pad_mask = create_padding_mask(tgt, pad_token=pad_token)
            tgt_causal_mask = create_causal_mask(tgt.size(1)).to(src.device)
            tgt_mask = tgt_pad_mask & tgt_causal_mask

            # Decode and predict next token
            decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
            logits = model.project(decoder_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # Get the most probable token

            # Append predicted token to the target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if <END> token is generated
            if next_token.item() == end_token:
                break

        return tgt.squeeze(0).tolist()  # Return generated sequence

    # Example usage of inference
    src_example = torch.randint(0, src_vocab_size, (1, src_seq_len))  # Random source sequence
    generated_sequence = translate_sentence(model, src_example, src_vocab_size, tgt_vocab_size)
    print("Generated sequence:", generated_sequence)
