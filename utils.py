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

    _, seq_length = inputs.shape
    num_zeros = max_seq_length - seq_length

    padded_inputs = torch.nn.functional.pad(
        inputs, (0, num_zeros), mode="constant", value=0
    )
    return padded_inputs
