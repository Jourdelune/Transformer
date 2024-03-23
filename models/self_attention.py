import torch

import numpy as np
import torch.nn as nn


class SelfAttention(nn.Module):
    """The self attention equation from the transformer paper
    """
    
    def __init__(self) -> None:
        """Add the softmax layer to the self attention instance
        """
        
        super().__init__()
        
        self.__softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the scaled dot product attention.

        :param q: the query vector (batch_size, num_head, seq_length, dk)
        :type q: torch.Tensor
        :param k: the key vector (batch_size, num_head, seq_length, dk)
        :type k: torch.Tensor
        :param v: the value vector (batch_size, num_head, seq_length, dk)
        :type v: torch.Tensor
        :return: the output of the self attention
        :rtype: torch.Tensor
        """
        
        # (batch_size, num_head, seq_length, dk) * (batch_size, num_head, seq_length, dk)
        k_t = k.transpose(2, 3)

        eq = torch.matmul(q, k_t)/np.sqrt(
            q.shape[-1]
        )

        return torch.matmul(self.__softmax(eq), v)
