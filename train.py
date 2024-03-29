import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from hyperparameters import HyperParameters
from model import Transformer
from tokenizer import Tokenizer

torch.autograd.set_detect_anomaly(True)

model = Transformer(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.N_LAYERS.value,
    HyperParameters.HEADS.value,
    HyperParameters.D_FFN.value,
    HyperParameters.MAX_SEQ_LENGHT.value,
    HyperParameters.DROPOUT_RATE.value,
)

tok = Tokenizer(max_seq_length=HyperParameters.MAX_SEQ_LENGHT.value)

train_iter = Multi30k(split="train")
train_dataloader = DataLoader(
    train_iter, batch_size=HyperParameters.BATCH_SIZE.value, shuffle=True
)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()

EPOCHS = 10

steps = 0
for epoch in range(EPOCHS):
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        src_data = data[0]
        tgt_data = data[1]

        # tokenization
        src_data = [tok.encode(sentence) for sentence in src_data]
        tgt_data = [tok.encode(sentence) for sentence in tgt_data]
        
        # convert to tensor
        src_data = torch.stack([torch.tensor(sentence) for sentence in src_data])
        tgt_data = torch.stack([torch.tensor(sentence) for sentence in tgt_data])
        
        # prediction
        output = model(src_data, tgt_data)

        # reshape target
        trg = tgt_data.reshape(-1)

        # reshape output
        output = output.reshape(-1, output.shape[-1]) # (batch_size * max_seq_length, vocab_size)

        loss = loss_function(output, trg)
        loss.backward()
 
        optimizer.step()
        
        steps += 1

        break
    break
