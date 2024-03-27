import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class DecoderLayer(nn.Module):
    """The decoder layer
    """

    def __init__(self, num_head: int, dim_model: int, d_ffn: int, dropout_rate: int) -> None:
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

        self.__dropout1 = nn.Dropout(dropout_rate)
        self.__dropout2 = nn.Dropout(dropout_rate)
        self.__dropout3 = nn.Dropout(dropout_rate)

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

        # dropout
        tgt_attention = self.__dropout1(tgt_attention)

        # residual connexion
        tgt_attention += tgt

        # layer norm
        tgt_attention = self.__layer_norm1(tgt_attention)

        # multi head attention
        cross_tgt_attention = self.__multi_head_attention(tgt_attention, enc_out, enc_out, src_mask)

        # dropout
        cross_tgt_attention = self.__dropout2(cross_tgt_attention)

        # residual connexion
        tgt_attention += cross_tgt_attention

        # layer norm
        tgt_attention = self.__layer_norm2(tgt_attention)

        # feed forward network
        tgt_ffn = self.__ffn(tgt_attention)

        # dropout
        tgt_ffn = self.__dropout3(tgt_ffn)

        # residual connexion
        tgt_attention += tgt_ffn

        # layer norm
        tgt_attention = self.__layer_norm3(tgt_attention)

        return tgt_attention
