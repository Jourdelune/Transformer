import torch

import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_seq_len: int =200):
        super().__init__()
        # transform the row vector to column vector
        position = torch.arange(max_seq_len).unsqueeze(1)

        # (nb token, dim)

        position = position.unsqueeze(1)

        print(position.shape)