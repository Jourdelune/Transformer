from enum import Enum


class HyperParameters(Enum):
    """
    Class that represent all hyperparameters of the model
    """

    VOCAB_SIZE: int = 10_000
    DIM_MODEL: int = 512  # has to be even
    MAX_SEQ_LENGHT: int = 256
    N_LAYERS: int = 6
    BATCH_SIZE: int = 32
    HEADS: int = 8 # can't be bigger than DIM_MODEL
    D_FFN: int = 2048
    DROPOUT_RATE: int = 0.1
