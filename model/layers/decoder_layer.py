import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class DecoderLayer(nn.Module):
    """The decoder layer
    """

    def __init__(self, num_head: int, dim_model: int, d_ffn: int) -> None:
        """Initialize the decoder layer

        :param num_head: the number of head for the attention layer
        :type num_head: int
        :param dim_model: the dimension of the embedding
        :type dim_model: int
        :param d_ffn: the dimension of the feed forward network
        :type d_ffn: int
        """

        super().__init__()

        self.__mask_multi_head_attention = MultiHeadAttention(num_head, dim_model)
        self.__multi_head_attention = MultiHeadAttention(num_head, dim_model) # also called cross attention

        self.__layer_norm1 = nn.LayerNorm(dim_model)
        self.__layer_norm2 = nn.LayerNorm(dim_model)
        self.__layer_norm3 = nn.LayerNorm(dim_model)

        self.__ffn = PositionWiseFeedForwardNetwork(dim_model, d_ffn)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """run a pass throught the layer

        :param tgt: the current prediction of the model
        :type tgt: torch.Tensor
        :param enc_out: the encoder output
        :type enc_out: torch.Tensor
        :param src_mask: the source mask of the input pass througth the encoder layer, defaults to None
        :type src_mask: torch.Tensor, optional
        :param tgt_mask: the target mask to make prediction not based on next word, defaults to None
        :type tgt_mask: torch.Tensor, optional
        :return: the attention output of dim (batch size, max seq length, word dim)
        :rtype: torch.Tensor
        """

        tgt_attention = self.__mask_multi_head_attention(tgt, tgt, tgt, tgt_mask)
        tgt += tgt_attention

        tgt = self.__layer_norm1(tgt)

        tgt_attention = self.__multi_head_attention(tgt, enc_out, enc_out, src_mask)
        tgt += tgt_attention

        tgt = self.__layer_norm2(tgt)

        tgt_ffn = self.__ffn(tgt)
        tgt += tgt_ffn

        return self.__layer_norm3(tgt)
