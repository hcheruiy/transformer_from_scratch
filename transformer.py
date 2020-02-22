
# imports from standard library
import math
import copy

# PyTorch
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.modules import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class Transformer1(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):
        super(Transformer1, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                   nhead = nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model = d_model,
                                                   nhead = nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
    def forward(self, src: Tensor, tgt: Tensor) -> None:
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        return output
    
class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention is All You Need"
    Expects input of size (N, T, E)
    Generates positional encoding of size (T, E), and adds this to each batch
    element.
    """

    def __init__(self, num_features: int, seq_len: int) -> None:
        super().__init__()

        # Encoding for each element is (seq_len x num_features)
        positional_encoding = torch.zeros(seq_len, num_features, requires_grad=False)

        # Generate position - list from 0 to seq_len
        # Reshape to (seq_len x 1)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        # These will be divided by
        #
        # (10000 ^ (i / num_features))
        #
        # where i is the dim
        #
        # So, we'll have one feature where the position is divided by 1, giving a
        # sine/cosine wave with frequency 2 * pi
        #
        # At the other extreme, we'll have a feature where the position is divided by
        # 10000, giving sine/cosine waves with frequency 2 * pi * 10000
        #
        # Another way of saying this is that this will be *multiplied* by
        # ((1/10000) ^ (i / num_features))
        #
        # or by
        #
        # exp ( log (1/10000) ^ (i / num_features) )
        #
        # or equivalently
        #
        # exp ( (i / num_features) * -log(10000) )
        div_term = torch.exp(
            (torch.arange(0, num_features, 2).float() / num_features)
            * math.log(1 / 10000)
        )
        # Now we alternate applying sine to these features vs. cosine
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add a first dimension of size 1
        # [seq_len x num_features] -> [1 x seq_len x num_features]
        positional_encoding = positional_encoding.unsqueeze(0)

        # Transpose to put sequence length in first position, batch size in second
        positional_encoding = positional_encoding.transpose(0, 1)

        # de-mean
        # due to all the cosine terms starting at 1, and the sine terms starting at
        # 0, the mean of these positional encodings is much greater than 0; adding
        # an embedding that is shifted like this seems sub optimal, so we'll
        # "de-mean" this matrix:
        positional_encoding = positional_encoding - positional_encoding.mean()

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encoding
