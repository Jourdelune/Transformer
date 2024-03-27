import torch
import torch.nn as nn


class PositionWiseFeedForwardNetwork(nn.Module):
    """the position wise feed forward network layer from transformers"""

    def __init__(self, dim_model: int, dim_ffn: int) -> None:
        """initialize default value

        :param dim_model: the dim of the input
        :type dim_model: int
        :param dim_ffn: the dim of the feed forward network
        :type dim_ffn: int
        """

        super().__init__()

        self.__ffn1 = nn.Linear(dim_model, dim_ffn)
        self.__ffn2 = nn.Linear(dim_ffn, dim_model)
        self.__relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """run a pass over the layer

        :param inputs: the input value
        :type inputs: torch.Tensor
        :return: the output value
        :rtype: torch.Tensor
        """

        # feed forward
        inputs = self.__ffn1(inputs)

        # relu
        inputs = self.__relu(inputs)

        # second feed forward
        inputs = self.__ffn2(inputs)

        return inputs
