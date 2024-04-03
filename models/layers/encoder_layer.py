import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class EncoderLayer(nn.Module):
    """Encoder layer of the transformer network."""

    def __init__(
        self, num_head: int, dim_model: int, d_ffn: int, dropout_rate: int
    ) -> None:
        """initialize the encoder layer

        :param num_head: the number of head in the multi head attention
        :type num_head: int
        :param dim_model: the dimension of the model
        :type dim_model: int
        :param d_ffn: the number of neurons in the feed forward network
        :type d_ffn: int
        :param dropout_rate: the dropout rate
        :type dropout_rate: int
        """

        super().__init__()

        self.__multi_head_attention = MultiHeadAttention(num_head, dim_model)
       
        # two different layer because there are learnable parameters
        self.__layer_norm1 = nn.LayerNorm(dim_model)
        self.__layer_norm2 = nn.LayerNorm(dim_model)
      
        self.__ffn = PositionWiseFeedForwardNetwork(dim_model, d_ffn)

        self.__dropout1 = nn.Dropout(dropout_rate)
        self.__dropout2 = nn.Dropout(dropout_rate)

    def forward(
        self, inputs: torch.Tensor, pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Run the input tensor through the encoder layer

        :param inputs: the input tensor
        :type inputs: torch.Tensor
        :param pad_mask: the padding mask tensor, defaults to None
        :type pad_mask: torch.Tensor, optional
        :return: the output tensor
        :rtype: torch.Tensor
        """
        
        # multi head attention
        encoder_val = self.__multi_head_attention(inputs, inputs, inputs, pad_mask)
        
        # regulate
        encoder_val = self.__dropout1(encoder_val)
        
        # residual connexion
        encoder_val += inputs

        # normalization over dim
        encoder_val = self.__layer_norm1(encoder_val)
        
        # feed forward outputs
        ffn_outputs = self.__ffn(encoder_val)

        # regulate
        encoder_val = self.__dropout2(
            encoder_val
        )  # can be dropout1 because there is no learning param in the dropout layer

        # residual connexions
        encoder_val += ffn_outputs

        # normalization
        return self.__layer_norm2(encoder_val)
