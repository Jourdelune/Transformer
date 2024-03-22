import torch

import torch.nn as nn


class SelfAttention():
    """The self attention equation from the transformer paper (not tested)
    """
    
    def __init__(self) -> None:
        """Add the softmax layer to the self attention instance
        """
        
        super().__init__()
        
        self.__softmax = torch.softmax()
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the scaled dot product attention.

        :param q: the query vector
        :type q: torch.Tensor
        :param k: the key vector
        :type k: torch.Tensor
        :param v: the value vector
        :type v: torch.Tensor
        :return: the output of the self attention
        :rtype: torch.Tensor
        """
        
        eq = torch.dot(q, k.T)/torch.sqrt(q.shape[-1])
        
        return torch.dot(self.__softmax(eq), v)
