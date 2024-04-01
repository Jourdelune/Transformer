import torch


def pad_tensor(tensor: torch.Tensor, max_seq_length: int, pad_token: int) -> torch.Tensor:
    """Function that pad the tensor by pad_token

    :param tensor: the tensor to pad
    :type tensor: torch.Tensor
    :param max_seq_length: the maximum lenght of the sentence
    :type max_seq_length: int
    :param pad_token: the token used in the padding
    :type pad_token: int
    :return: a padded tensor
    :rtype: torch.Tensor
    """
    
    return torch.cat(
        (
            tensor[:max_seq_length],
            torch.tensor([pad_token] * (max_seq_length - tensor.shape[0]), dtype=torch.int64)
        )
    )
