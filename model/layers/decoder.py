import torch
import torch.nn as nn

from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """the decoder block
    """

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        ffn_val: int,
        vocab_size: int,
        dropout_rate: int
    ) -> None:
        """Initialize the value for the decoder block

        :param dim_model: the dim of the word
        :type dim_model: int
        :param num_layers: the number of decoder layer
        :type num_layers: int
        :param num_heads: the number of head for the self attention layer
        :type num_heads: int
        :param ffn_val: the number of dim of the feed forward network
        :type ffn_val: int
        :param vocab_size: the size of the vocab
        :type vocab_size: int
        """

        super().__init__()

        self.__decoders = nn.ModuleList(
            [DecoderLayer(num_heads, dim_model, ffn_val, dropout_rate) for _ in range(num_layers)]
        )

        self.__ffn = nn.Linear(dim_model, vocab_size)
        self.__softmax = nn.Softmax(2)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """run a pass through the decoder block 

        :param tgt: the current decoder output
        :type tgt: torch.Tensor
        :param enc_out: the encoder output
        :type enc_out: torch.Tensor
        :param src_mask: the source mask of the input sentence that went through the encoder block, defaults to None
        :type src_mask: torch.Tensor, optional
        :param tgt_mask: the target mask (current prediction of the model), defaults to None
        :type tgt_mask: torch.Tensor, optional
        :return: a output that for each position in the sequence return the probability of each word.
        :rtype: torch.Tensor
        """

        for decoder in self.__decoders:
            tgt = decoder(tgt, enc_out, src_mask, tgt_mask)

        tgt = self.__ffn(tgt)

        return self.__softmax(tgt)
