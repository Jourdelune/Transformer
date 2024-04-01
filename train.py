from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from hyperparameters import HyperParameters
from models import Transformer
from tokenizer import Tokenizer
from utils import pad_tensor


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform a list of sample into batch

    :param batch: a list of sentences
    :type batch: list
    :return: a tuple contain the batch
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sample = tok.string_to_vocab(src_sample, src=True)
        tgt_sample = tok.string_to_vocab(tgt_sample, src=False)

        src_sample = pad_tensor(src_sample, HyperParameters.MAX_SEQ_LENGHT.value, 1)
        tgt_sample = pad_tensor(tgt_sample, HyperParameters.MAX_SEQ_LENGHT.value, 1)

        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    return src_batch, tgt_batch


data = Multi30k(split="train")
train_dataloader = DataLoader(
    data, batch_size=HyperParameters.BATCH_SIZE.value, collate_fn=collate_fn
)

tok = Tokenizer(
    "de_core_news_sm", "en_core_web_sm", HyperParameters.VOCAB_SIZE.value, data
)
model = Transformer(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.MAX_SEQ_LENGHT.value,
    HyperParameters.DROPOUT_RATE.value
)

for src, tgt in train_dataloader:
    # print(tok.vocab_to_string(src[0,], src=True))
    # print(tok.vocab_to_string(tgt[0,], src=False))
    
    out = model(src, tgt)
    print(out[0][0])
    break
