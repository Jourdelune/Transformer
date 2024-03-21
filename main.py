import torch

from models.embeddings import Embeddings
from models.positional_encoding import PositionalEncoding
from models.tokenizer import Tokenizer

from hyperparameters import HyperParameters
from utils import *


torch.manual_seed(1)


x = torch.randint(
    0,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value),
)

embedding_layer = Embeddings(
    HyperParameters.VOCAB_SIZE.value, HyperParameters.DIM_MODEL.value
)

print(embedding_layer(x).shape)
