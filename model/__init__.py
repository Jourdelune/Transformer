from typing import Tuple

import torch
import torch.nn as nn

from .blocks.decoder import Decoder
from .layers.embeddings import Embeddings
from .blocks.encoder import Encoder
from .layers.positional_encoding import PositionalEncoding
from .utils import pad_tensor


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
        dropout_rate: int,
    ) -> None:
        """Initialize the transformer model

        :param vocab_size: the number of words in the vocabulary
        :type vocab_size: int
        :param dim_model: the dimension of the word embedding
        :type dim_model: int
        :param num_layers: the number of layers in the encoder and decoder
        :type num_layers: int
        :param num_heads: the number of heads in the multi-head attention
        :type num_heads: int
        :param ffn_val: the number of neurons in the feed forward network
        :type ffn_val: int
        :param max_seq_length: the maximum length of the sequence
        :type max_seq_length: int
        :param dropout_rate: the dropout rate
        :type dropout_rate: int
        """

        super().__init__()

        self.__encoder = Encoder(
            dim_model, num_layers, num_heads, ffn_val, dropout_rate
        )
        self.__decoder = Decoder(
            dim_model, num_layers, num_heads, ffn_val, vocab_size, dropout_rate
        )

        self.__embedding_layer = Embeddings(vocab_size, dim_model)
        self.__positionnal_encoder = PositionalEncoding(dim_model, vocab_size)

        self.__max_seq_length = max_seq_length

        self.__dropout1 = nn.Dropout(dropout_rate)
        self.__dropout2 = nn.Dropout(dropout_rate)

    @staticmethod
    def __generate_mask(
        src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to generate source mask and look ahead mask

        :param src: the source padded sentence
        :type src: torch.Tensor
        :param tgt: the target padded sentence
        :type tgt: torch.Tensor
        :return: a tuple that contain the source and the target sentences mask
        :rtype: tuple
        """

        src_mask = (src == 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt == 0).unsqueeze(1).unsqueeze(2)

        seq_length = tgt.size(1)
        nopeak_mask = torch.triu(
            torch.ones(1, seq_length, seq_length), diagonal=1
        ).bool()

        tgt_mask = tgt_mask | nopeak_mask

        return src_mask, tgt_mask

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer model

        :param source: the source padded sentence
        :type source: torch.Tensor
        :param target: the target padded sentence
        :type target: torch.Tensor
        :return: the output of the transformer model
        :rtype: torch.Tensor
        """

        source = pad_tensor(source, self.__max_seq_length)
        target = pad_tensor(target, self.__max_seq_length)

        mask_src, mask_tgt = self.__generate_mask(source, target)

        source = self.__embedding_layer(source)
        source = self.__positionnal_encoder(source)
        source = self.__dropout1(source)

        target = self.__embedding_layer(target)
        target = self.__positionnal_encoder(target)
        target = self.__dropout2(target)

        encoder_outputs = self.__encoder(source, pad_mask=mask_src)
        decoder_outputs = self.__decoder(target, encoder_outputs, mask_src, mask_tgt)

        return decoder_outputs
