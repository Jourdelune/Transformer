import torch

from models.layers.embeddings import Embeddings
from models.layers.positional_encoding import PositionalEncoding
from models.layers.encoder import Encoder
from models.tokenizer import Tokenizer

from hyperparameters import HyperParameters
from utils import *


x = torch.randint(
    0,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value-2),
)

x = pad_inputs(x, HyperParameters.MAX_SEQ_LENGHT.value)

embedding_layer = Embeddings(
    HyperParameters.VOCAB_SIZE.value, HyperParameters.DIM_MODEL.value
)

positionnal_encoder = PositionalEncoding(
    HyperParameters.DIM_MODEL.value, HyperParameters.MAX_SEQ_LENGHT.value
)


embed = embedding_layer(x)*0
x = positionnal_encoder(embed)

encoder = Encoder(
    HyperParameters.HEADS.value, HyperParameters.DIM_MODEL.value, HyperParameters.D_FFN.value
)

a = encoder(x)

