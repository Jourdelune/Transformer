import torch


def convert_token_to_idx(tokens: list) -> torch.tensor:
    """Function that transform a list of index of words to an array of vector of 1 or 0.

    :param tokens: the list of tokens.
    :type tokens: list
    :rtype: torch.tensor
    """

    out_tensors = torch.tensor([idx for idx in tokens])

    return out_tensors


def pad_inputs(inputs: torch.Tensor, max_seq_length: int) -> torch.Tensor:
    """Function that add padding value to match the max seq length

    :param inputs: the inputs tensor
    :type inputs: torch.Tensor
    :param max_seq_length: the max number of sequence
    :type max_seq_length: int
    :return: a padded matrix
    :rtype: torch.Tensor
    """
    
    batch_size, seq_length, dim_embed = inputs.shape
    target = torch.zeros(batch_size, max_seq_length, dim_embed)
    target[:, :seq_length, :] = inputs

    return target
