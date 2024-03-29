import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from hyperparameters import HyperParameters
from model import Transformer
from tokenizer import Tokenizer

torch.autograd.set_detect_anomaly(True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def build_model():
    model = Transformer(
        HyperParameters.VOCAB_SIZE.value,
        HyperParameters.DIM_MODEL.value,
        HyperParameters.N_LAYERS.value,
        HyperParameters.HEADS.value,
        HyperParameters.D_FFN.value,
        HyperParameters.MAX_SEQ_LENGHT.value,
        HyperParameters.DROPOUT_RATE.value,
        device
    )

    model.to(device)

    return model


def train(model, train_dataloader, EPOCHS):
    loss_function = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()
    steps = 0

    for epoch in range(EPOCHS):
        for _, data in enumerate(train_dataloader):
            src_data = data[0]
            tgt_data = data[1]

            # tokenization
            src_data = [tok.encode(sentence) for sentence in src_data]
            tgt_data = [tok.encode(sentence) for sentence in tgt_data]

            # convert to tensor
            src_data = torch.stack([torch.tensor(sentence)
                                   for sentence in src_data])
            tgt_data = torch.stack([torch.tensor(sentence)
                                   for sentence in tgt_data])

            optimizer.zero_grad()
            
            # prediction
            output = model(src_data, tgt_data[:, :-1])

            # reshape target
            trg = tgt_data.reshape(-1)

            # reshape output
            output = output.reshape(
                -1, output.shape[-1]
            )  # (batch_size * max_seq_length, vocab_size)

            loss = loss_function(output.cuda(), trg.cuda())
            loss.backward()

            optimizer.step()

            steps += 1

            print(
                f"Epoch {epoch + 1}/{EPOCHS} - Step {steps} - Loss {loss.item()}")

        torch.save(model.state_dict(), 'weights/model_weights')


def build_dataset():
    train_iter = Multi30k(split="train")
    train_dataloader = DataLoader(
        train_iter, batch_size=HyperParameters.BATCH_SIZE.value, shuffle=True
    )

    return train_dataloader


tok = Tokenizer(max_seq_length=HyperParameters.MAX_SEQ_LENGHT.value)

model = build_model()
dataset = build_dataset()

train(model, dataset, 1)
