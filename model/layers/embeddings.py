import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Embedding class that transform a list of one-hot vector representing the index of
    a word in the vocabulary to a representation in a vectorized space
    """

    def __init__(self, vocab_size: int, dim_model: int) -> None:
        """A method that initializes the attributes of the Embedding class

        :param vocab_size: the number of words in the vocabulary
        :type vocab_size: int
        :param dim_model: the dimension of a vector that will contain the feature to represent a word
        :type dim_model: int
        """
        super().__init__()

        self.__embeddings = nn.Embedding(vocab_size, dim_model)
        self.__dim_model = torch.tensor(dim_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pass a tensors of dim (batch_size, max_seq_length) to the embedding layer

        :param inputs: the input that contains the words
        :type inputs: torch.Tensor
        :return: an output of dim (batch_size, max_seq_length, dim_model)
        :rtype: torch.Tensor
        """

        return self.__embeddings(inputs) * torch.sqrt(self.__dim_model)
