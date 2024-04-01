import torch
import torch.nn as nn


class Encoder(nn.Module):
    """The encoder block"""

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
        dropout_rate: int,
    ) -> None:
        """initialize the encoder block

        :param dim_model: the dimension of the model
        :type dim_model: int
        :param num_layers: the number of layers
        :type num_layers: int
        :param num_heads: the number of heads
        :type num_heads: int
        :param ffn_val: the number of neurons in the feed forward network
        :type ffn_val: int
        :param dropout_rate: the dropout rate
        :type dropout_rate: int
        """

        super().__init__()

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """calculate the output of the encoder block

        :param src: the value that the encoders layers will process of shape (bias, seq_length)
        :type src: torch.Tensor
        :param src_mask: the mask of the current value of shape (bias, 1, 1, seq_length), defaults to None
        :type src_mask: torch.Tensor, optional
        :return: the predictions of the encoder block
        :rtype: torch.Tensor
        """

        return src
