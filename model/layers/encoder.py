import torch
import torch.nn as nn

from .embeddings import Embeddings
from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
    ) -> None:
        super().__init__()

        self.__embedding_layer = Embeddings(vocab_size, dim_model)
        self.__positionnal_encoder = PositionalEncoding(dim_model, vocab_size)

        self.__encoders = nn.ModuleList(
            [EncoderLayer(num_heads, dim_model, ffn_val) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.__embedding_layer(inputs) * 0
        inputs = self.__positionnal_encoder(inputs)

        for encoder in self.__encoders:
            inputs = encoder(inputs)

        return inputs
