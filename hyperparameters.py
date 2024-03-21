from enum import Enum


class HyperParameters(Enum):
    """
    Class that represent all hyperparameters of the model
    """
    
    VOCAB_SIZE: int = 10_000
    DIM_MODEL: int = 512
    MAX_SEQ_LENGHT: int = 256
    BATCH_SIZE: int = 128
