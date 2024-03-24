import torch

from hyperparameters import HyperParameters
from model.layers.encoder import Encoder
from utils import *

x = torch.randint(
    0,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value - 2),
)

x = pad_inputs(x, HyperParameters.MAX_SEQ_LENGHT.value)

encoder = Encoder(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.N_LAYERS.value,
    HyperParameters.HEADS.value,
    HyperParameters.D_FFN.value,
)

out = encoder(x)
print(out)
