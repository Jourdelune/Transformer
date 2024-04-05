import torch
import torch.nn as nn

from ..layers.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """the decoder block"""

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
        vocab_size: int,
        dropout_rate: int,
    ) -> None:
        """initialize the decoder block

        :param dim_model: the dimension of the model
        :type dim_model: int
        :param num_layers: the number of layers
        :type num_layers: int
        :param num_heads: the number of heads
        :type num_heads: int
        :param ffn_val: the number of neurons in the feed forward network
        :type ffn_val: int
        :param vocab_size: the size of the vocabulary
        :type vocab_size: int
        :param dropout_rate: the dropout rate
        :type dropout_rate: int
        """

        super().__init__()

        self.__decoders = nn.ModuleList(
            [
                DecoderLayer(num_heads, dim_model, ffn_val, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        self.__ffn = nn.Linear(dim_model, vocab_size)
        self.__softmax = nn.Softmax(2)

    def forward(
        self,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        look_ahead_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the input tensor through the decoder layers

        :param enc_output: the output of the encoder block
        :type enc_output: torch.Tensor
        :param src_mask: the mask used in the encoder block
        :type src_mask: torch.Tensor
        :param tgt: the current prediction of the model
        :type tgt: torch.Tensor
        :param look_ahead_mask: the mask used in the masked multi head attention layer
        :type look_ahead_mask: torch.Tensor
        :return: rhe output of the decoder block
        :rtype: torch.Tensor
        """

        for decoder in self.__decoders:
            tgt = decoder(enc_output, src_mask, tgt, look_ahead_mask)
        
        tgt = self.__ffn(tgt)

        return tgt
