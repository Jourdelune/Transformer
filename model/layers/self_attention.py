import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """The self attention equation from the transformer paper"""

    def __init__(self) -> None:
        """Add the softmax layer to the self attention instance"""

        super().__init__()

        self.__softmax = nn.Softmax(
            dim=-1
        )  # calculate dim on the last dim of the input vector

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply the scaled dot product attention.

        :param q: the query vector (batch_size, num_head, seq_length, dk)
        :type q: torch.Tensor
        :param k: the key vector (batch_size, num_head, seq_length, dk)
        :type k: torch.Tensor
        :param v: the value vector (batch_size, num_head, seq_length, dk)
        :type v: torch.Tensor
        :param pad_mask: the padding mask
        :type pad_mask: torch.Tensor
        :return: the output of the self attention
        :rtype: torch.Tensor
        """

        # (batch_size, num_head, seq_length, dk) to (batch_size, num_head, dk, seq_length)
        k_t = k.transpose(2, 3)

        eq = torch.matmul(q, k_t) / np.sqrt(q.shape[-1])

        if pad_mask is not None:
            eq = eq.masked_fill(
                pad_mask, -1e10
            )  # ty the broadcast, don't forget to update the eq matrix lol

        return torch.matmul(self.__softmax(eq), v)
