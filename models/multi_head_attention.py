import torch

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Apply the multi head attention layer
    """
    
    def __init__(self, num_head: int, dim_model: int) -> None:
        """Initialize the linear q, k and v learnable parameters.

        :param num_head: the number of head (self attention) layer that contain the multi head attention layer
        :type num_head: int
        :param dim_model: the dim of the input of the model
        :type dim_model: int
        """
        
        super().__init__()
        
        self.__dk = self.__dv = dim_model//num_head
        self.__num_head = num_head
        
        self.__Q = nn.Linear(dim_model, dim_model)
        self.__K = nn.Linear(dim_model, dim_model)
        self.__V = nn.Linear(dim_model, dim_model)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run a pass througth the multi head attention layer

        :param q: the query tensor
        :type q: torch.Tensor
        :param k: the key tensor
        :type k: torch.Tensor
        :param v: the value tensor
        :type v: torch.Tensor
        :return: the output of the calculated attention.
        :rtype: torch.Tensor
        """
        
        q = self.__Q(q)
        k = self.__K(k)
        v = self.__V(v)
      
        # reshape from (batch_size, seq_length, dim_model) to (batch_size, num_head, seq_length, dk).
        q = q.view(q.shape[0], self.__num_head, q.shape[1], self.__dk)
        k = k.view(v.shape[0], self.__num_head, v.shape[1], self.__dk)
        v = v.view(v.shape[0], self.__num_head, v.shape[1], self.__dv)
        
        print(q.shape, k.shape, v.shape)