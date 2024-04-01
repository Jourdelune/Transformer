import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Layer that add the positionnal encoding"""

    def __init__(self, dim_model: int, max_seq_len: int, device: str = "cpu"):
        """A method that initializes the attributes of the PositionalEncoding class

        :param dim_model: the dimension of a vector that will contain the feature to represent a word
        :type dim_model: int
        :param max_seq_len: the maximum lenght of a sentence
        :type max_seq_len: int
        :param device: the device to use, defaults to "cpu"
        :type device: str, optional
        """

        super().__init__()

        pe = torch.zeros((1, max_seq_len, dim_model))

        pos = torch.arange(max_seq_len).unsqueeze(1)
        divid = 10_000 ** (torch.arange(0, dim_model, 2) / dim_model)

        pe[0, :, 0::2] = torch.sin(pos / divid)
        pe[0, :, 1::2] = torch.cos(pos / divid)

        self.__pe = pe.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Function that add to the inputs the positionnal encoding

        :param inputs: the embedding tensor of dim (batch_sizes, seq length, dim_size)
        :type inputs: torch.Tensor
        :return: the inputs with pos encoding
        :rtype: torch.Tensor
        """

        return inputs + self.__pe[:, : inputs.shape[1], :]
