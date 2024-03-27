import torch
import torch.nn as nn

from hyperparameters import HyperParameters
from model import Transformer


transformer = Transformer(
    HyperParameters.VOCAB_SIZE.value,
    HyperParameters.DIM_MODEL.value,
    HyperParameters.N_LAYERS.value,
    HyperParameters.HEADS.value,
    HyperParameters.D_FFN.value,
    HyperParameters.MAX_SEQ_LENGHT.value,
    HyperParameters.DROPOUT_RATE.value,
)

# TODO
