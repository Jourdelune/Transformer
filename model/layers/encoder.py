import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """The stack of all the encoder layer."""

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
    ) -> None:
        """Stack all encoders layers

        :param dim_model: the dim of the model
        :type dim_model: int
        :param num_layers: the number of encoder layer to stack
        :type num_layers: int
        :param num_heads: the number of head used in the attention layer
        :type num_heads: int
        :param ffn_val: the dim of the feed forward layer
        :type ffn_val: int
        """

        super().__init__()

        self.__encoders = nn.ModuleList(
            [EncoderLayer(num_heads, dim_model, ffn_val) for _ in range(num_layers)]
        )

    def forward(
        self, inputs: torch.Tensor, pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """run a pass througth all the encoders layers

        :param inputs: the inputs values
        :type inputs: torch.Tensor
        :param pad_mask: the padding for the self attention layer
        :type pad_mask: torch.Tensor
        :return: the outputs values
        :rtype: torch.Tensor
        """

        for encoder in self.__encoders:
            inputs = encoder(inputs, pad_mask)

        return inputs
