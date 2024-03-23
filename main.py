import torch

from models.embeddings import Embeddings
from models.multi_head_attention import MultiHeadAttention
from models.positional_encoding import PositionalEncoding
from models.tokenizer import Tokenizer

from hyperparameters import HyperParameters
from utils import *


x = torch.randint(
    0,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value),
)


embedding_layer = Embeddings(
    HyperParameters.VOCAB_SIZE.value, HyperParameters.DIM_MODEL.value
)

positionnal_encoder = PositionalEncoding(
    HyperParameters.DIM_MODEL.value, HyperParameters.MAX_SEQ_LENGHT.value
)

multi_head_attention = MultiHeadAttention(
    HyperParameters.HEADS.value, HyperParameters.DIM_MODEL.value
)

embed = embedding_layer(x)*0
x = positionnal_encoder(embed)

multi = multi_head_attention(x, x, x)

print(multi)
