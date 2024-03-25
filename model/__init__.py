import torch
import torch.nn as nn

from .layers.embeddings import Embeddings
from .layers.encoder import Encoder
from .layers.positional_encoding import PositionalEncoding
from .utils import pad_inputs


class Transformer(nn.Module):
    """The transformer model"""

    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
        max_seq_length: int,
    ) -> None:
        """initialize the hyperparmeters of the transformer model

        :param vocab_size: the number of token in the vocabulary
        :type vocab_size: int
        :param dim_model: the dim of the model (dim of the embedding layer)
        :type dim_model: int
        :param num_layers: the number of encoder & decoder layer
        :type num_layers: int
        :param num_heads: the number of head used in the attention layer
        :type num_heads: int
        :param ffn_val: the feed forward dimension
        :type ffn_val: int
        """

        super().__init__()

        self.__encoder = Encoder(dim_model, num_layers, num_heads, ffn_val)

        self.__embedding_layer = Embeddings(vocab_size, dim_model)
        self.__positionnal_encoder = PositionalEncoding(dim_model, vocab_size)

        self.__max_seq_length = max_seq_length

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """return the outputs througth the transformer model

        :param inputs: the inputs batched sentences
        :type inputs: torch.Tensor
        :return: the outputs from the transformer model
        :rtype: torch.Tensor
        """
        inputs, pad_mask = pad_inputs(inputs, self.__max_seq_length)
        inputs = self.__embedding_layer(inputs) * 0
        inputs = self.__positionnal_encoder(inputs)

        encoder_outputs = self.__encoder(inputs, pad_mask=pad_mask)

        return encoder_outputs
