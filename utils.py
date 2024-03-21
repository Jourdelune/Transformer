import torch


def convert_token_to_idx(tokens: list) -> torch.tensor:
    """Function that transform a list of index of words to an array of vector of 1 or 0.

    :param tokens: the list of tokens.
    :type tokens: list
    :rtype: torch.tensor
    """

    out_tensors = torch.tensor([idx for idx in tokens])

    return out_tensors
