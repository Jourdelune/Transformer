import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class DecoderLayer(nn.Module):
    """The decoder layer"""

    def __init__(
        self, num_head: int, dim_model: int, d_ffn: int, dropout_rate: int
    ) -> None:
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
        self.__multi_head_attention = MultiHeadAttention(
            num_head, dim_model
        )  # also called cross attention

        self.__layer_norm1 = nn.LayerNorm(dim_model)
        self.__layer_norm2 = nn.LayerNorm(dim_model)
        self.__layer_norm3 = nn.LayerNorm(dim_model)

        self.__ffn = PositionWiseFeedForwardNetwork(dim_model, d_ffn)

        self.__dropout1 = nn.Dropout(dropout_rate)
        self.__dropout2 = nn.Dropout(dropout_rate)
        self.__dropout3 = nn.Dropout(dropout_rate)

    def forward(
        self,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        look_ahead_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the input tensor through the decoder layer

        :param enc_output: the output of the encoder
        :type enc_output: torch.Tensor
        :param src_mask: the mask for the source tensor, defaults to None
        :type src_mask: torch.Tensor
        :param tgt: the target tensor (current output of the decoder)
        :type tgt: torch.Tensor
        :param look_ahead_mask: the mask for the target tensor, defaults to None
        :type look_ahead_mask: torch.Tensor
        :return: the output tensor
        :rtype: torch.Tensor
        """
        
        __tgt = tgt
        out = self.__mask_multi_head_attention(tgt, tgt, tgt, look_ahead_mask)
        
        # dropout & layer norm
        out = self.__dropout1(out)
        out = self.__layer_norm1(out + __tgt)

        __out = out

        out = self.__multi_head_attention(out, enc_output, enc_output, src_mask)

        # dropout
        out = self.__dropout2(out)

        # residual connexion & layer norm
        out = self.__layer_norm2(out + __out)

        __out = out
        
        # feed forward network
        out = self.__ffn(out)
       
        # dropout & residual connexion & layer norm
        out = self.__dropout3(out)

        out = self.__layer_norm3(out + __out)
      
        return out
