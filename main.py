import torch

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


src_data = torch.randint(
    1,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value - 2),
)

tgt_data = torch.randint(
    1,
    HyperParameters.VOCAB_SIZE.value,
    (HyperParameters.BATCH_SIZE.value, HyperParameters.MAX_SEQ_LENGHT.value - 2),
)
out = transformer(src_data, tgt_data)
