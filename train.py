from typing import Tuple

import torch
import torch.nn as nn
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

    src_batch, tgt_batch, label_batch = [], [], []
    for src_sample, tgt_sample in batch:
        src_sample = tok.string_to_vocab(src_sample, src=True)
        tgt_sample = tok.string_to_vocab(tgt_sample, src=False)
        label_sample = tok.create_label(tgt_sample)

        src_sample = pad_tensor(
            src_sample, HyperParameters.MAX_SEQ_LENGHT.value, 1)
        tgt_sample = pad_tensor(
            tgt_sample, HyperParameters.MAX_SEQ_LENGHT.value, 1)
        label_sample = pad_tensor(
            label_sample, HyperParameters.MAX_SEQ_LENGHT.value, 1)

        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        label_batch.append(label_sample)

    src_batch = torch.stack(src_batch).to(torch.long)
    tgt_batch = torch.stack(tgt_batch).to(torch.long)
    label_batch = torch.stack(label_batch).to(torch.long)

    return src_batch, tgt_batch, label_batch


data = Multi30k(split="train")
tok = Tokenizer(
    "de_core_news_sm", "en_core_web_sm", HyperParameters.VOCAB_SIZE.value, data
)
train_dataloader = DataLoader(
    data, batch_size=HyperParameters.BATCH_SIZE.value, collate_fn=collate_fn
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


model = Transformer(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.MAX_SEQ_LENGHT.value,
    HyperParameters.DROPOUT_RATE.value,
    HyperParameters.N_LAYERS.value,
    HyperParameters.HEADS.value,
    HyperParameters.D_FFN.value,
    device
).to(device)

for p in model.parameters():
    if p.dim() > 1 and True is False:
        nn.init.xavier_uniform_(p)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1).to(device)

steps = 0

for epoch in range(HyperParameters.EPOCH_NB.value):
    model.train()

    for idx, data in enumerate(train_dataloader):
        src, tgt, label = data
        
        src = src.to(device)
        tgt = tgt.to(device)
        label = label.to(device)

        steps += 1
     
        output = model(src, tgt)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(output.view(-1, output.shape[-1]), label.view(-1))

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad()

        print(
            f"Epoch {epoch + 1}/{HyperParameters.EPOCH_NB.value} - Step {steps} - Batch {idx} - Loss {loss.item()}")

torch.save(model.state_dict(), 'model.pt')
