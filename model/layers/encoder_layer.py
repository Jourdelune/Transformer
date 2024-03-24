import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class EncoderLayer(nn.Module):
    """Encoder layer of the transformer network.
    """

    def __init__(self, num_head: int, dim_model: int, d_ffn: int) -> None:
        """Construct the encoder layer

        :param num_head: the number of head of the multi head attention layer
        :type num_head: int
        :param dim_model: the dimension of the inputs of the model
        :type dim_model: int
        :param d_ffn: the dimension of the feed forwar network
        :type d_ffn: int
        """

        super().__init__()

        self.__multi_head_attention = MultiHeadAttention(num_head, dim_model)

        # two different layer because there are learnable parameters
        self.__layer_norm1 = nn.LayerNorm(dim_model)
        self.__layer_norm2 = nn.LayerNorm(dim_model)

        self.__ffn = PositionWiseFeedForwardNetwork(dim_model, d_ffn)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """run the encoder layer on inputs

        :param inputs: the inputs values
        :type inputs: torch.Tensor
        :return: the outputs values
        :rtype: torch.Tensor
        """

        # multi head attention
        encoder_val = self.__multi_head_attention(inputs, inputs, inputs)

        # residual connexions
        encoder_val += inputs

        # normalization over dim
        encoder_val = self.__layer_norm1(encoder_val)

        # feed forward outputs
        ffn_outputs = self.__ffn(encoder_val)

        # residual connexions
        encoder_val += ffn_outputs

        # normalization
        return self.__layer_norm2(encoder_val)
