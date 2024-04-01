from typing import Tuple

import torch
import torch.nn as nn

from .layers.embedings import Embeddings
from .layers.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    The Transformer Model
    """

    def __init__(self, vocab_size: int, dim_model: int, max_seq_len: int) -> None:
        super().__init__()

        self.__embeddings = Embeddings(vocab_size, dim_model)
        self.__pos_encoding = PositionalEncoding(dim_model, max_seq_len)

    @staticmethod
    def __generate_mask(
        src: torch.Tensor, tgt: torch.Tensor, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to generate source mask and look ahead mask

        :param src: the source text to pad of shape (bias, max_seq_length)
        :type src: torch.Tensor
        :param tgt: the target text to pad of shape (bias, max_seq_length)
        :type tgt: torch.Tensor
        :param device: the device to use, defaults to "cpu"
        :type device: str, optional
        :return: a tuple that contain the source and the target sentences mask
        :rtype: tuple
        """

        src_mask = (
            (src == 1).unsqueeze(1).unsqueeze(2)
        )  # (bias, n_head, 1, seq_lenght) -> because the dimension of the attention windows
        tgt_mask = (
            (tgt == 1).unsqueeze(1).unsqueeze(2)
        )  # (bias, n_head, seq_length, seq_lenght)

        seq_length = tgt.size(1)
        nopeak_mask = (
            torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
            .to(device)
            .bool()
        )

        tgt_mask = tgt_mask | nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.__generate_mask(src, tgt)

        src = self.__pos_encoding(self.__embeddings(src))
        tgt = self.__pos_encoding(self.__embeddings(tgt))

        return src
