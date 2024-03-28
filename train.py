import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from hyperparameters import HyperParameters
from model import Transformer
from tokenizer import Tokenizer

model = Transformer(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.N_LAYERS.value,
    HyperParameters.HEADS.value,
    HyperParameters.D_FFN.value,
    HyperParameters.MAX_SEQ_LENGHT.value,
    HyperParameters.DROPOUT_RATE.value,
)

tok = Tokenizer()

train_iter = Multi30k(split="train")
train_dataloader = DataLoader(
    train_iter, batch_size=HyperParameters.BATCH_SIZE.value, shuffle=True
)

model.train()

EPOCHS = 10

steps = 0
for epoch in range(EPOCHS):
    for i, data in enumerate(train_dataloader):
        src_data = data[0]
        tgt_data = data[1]

        steps += 1