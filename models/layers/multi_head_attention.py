import torch
import torch.nn as nn

from .self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    """Apply the multi head attention layer"""

    def __init__(self, num_head: int, dim_model: int) -> None:
        """Initialize the linear q, k and v learnable parameters.

        :param num_head: the number of head (self attention) layer that contain the multi head attention layer
        :type num_head: int
        :param dim_model: the dim of the input of the model
        :type dim_model: int
        """

        super().__init__()

        torch.manual_seed(0)

        self.__dk = self.__dv = dim_model // num_head
        self.__num_head = num_head

        self.__WQ = nn.Linear(dim_model, dim_model)
        self.__WK = nn.Linear(dim_model, dim_model)
        self.__WV = nn.Linear(dim_model, dim_model)

        self.__W0 = nn.Linear(dim_model, dim_model)

        self.__attention = SelfAttention()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run a pass througth the multi head attention layer

        :param q: the query tensor
        :type q: torch.Tensor
        :param k: the key tensor
        :type k: torch.Tensor
        :param v: the value tensor
        :type v: torch.Tensor
        :param pad_mask: the padding for the self attention layer
        :type pad_mask: torch.Tensor
        :return: the output of the calculated attention.
        :rtype: torch.Tensor
        """

        # calculate QW^q, QW^k, QW^v
        q = self.__WQ(q)
        k = self.__WK(k)
        v = self.__WV(v)

        old_shape = q.shape

        # reshape from (batch_size, seq_length, dim_model) to (batch_size, num_head, seq_length, dk) to get QW1, QW2 .. Qwhead
        q = q.reshape(q.shape[0], q.shape[1], self.__num_head, self.__dk).transpose(
            1, 2
        )
        k = k.reshape(k.shape[0], k.shape[1], self.__num_head, self.__dk).transpose(
            1, 2
        )
        v = v.reshape(v.shape[0], v.shape[1], self.__num_head, self.__dv).transpose(
            1, 2
        )

        attention_values = self.__attention(q, k, v, pad_mask)

        # concat all layer to return to (batch_size, seq_length, dim_model)
        attention_values = attention_values.transpose(1, 2).reshape(old_shape)

        return self.__W0(attention_values)
