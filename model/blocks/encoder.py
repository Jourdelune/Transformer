import torch
import torch.nn as nn

from ..layers.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """The stack of all the encoder layer."""

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

        self.__encoders = nn.ModuleList(
            [
                EncoderLayer(num_heads, dim_model, ffn_val, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Run the input tensor through the encoder layers

        :param inputs: the input tensor
        :type inputs: torch.Tensor
        :param pad_mask: the padding mask tensor, defaults to None
        :type pad_mask: torch.Tensor, optional
        :return: the output tensor
        :rtype: torch.Tensor
        """

        for encoder in self.__encoders:
            inputs = encoder(inputs, pad_mask)

        return inputs
