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
    (HyperParameters.MAX_SEQ_LENGHT.value, HyperParameters.BATCH_SIZE.value),
)


embedding_layer = Embeddings(
    HyperParameters.VOCAB_SIZE.value, HyperParameters.DIM_MODEL.value
)

positionnal_encoder = PositionalEncoding(
    HyperParameters.DIM_MODEL.value, HyperParameters.MAX_SEQ_LENGHT.value
)

embed = embedding_layer(x)*0
pos = positionnal_encoder(embed) # (seq length, batch size, model dim)
import matplotlib.pyplot as plt
import numpy as np

# Assuming pos is your positionnal encoding matrix
pos = pos.detach().numpy() # Convert to numpy array if it's a tensor

import matplotlib.pyplot as plt
import numpy as np


# Choose the first batch and first 'n' dimensions for visualization
n = 512  # You can change this value
batch = 0

# Transpose the matrix for heatmap
pos_batch = pos[:, batch, :n]

# Create a heatmap
plt.figure(figsize=(10, 8))
mappable = plt.imshow(pos_batch, cmap='viridis')

# Add labels
plt.xlabel('Sequence Length')
plt.ylabel('Model Dimension')
plt.title('Position Encoding Heatmap')

# Add colorbar
plt.colorbar(mappable)

# Save the plot as an image file
plt.savefig('position_encoding_heatmap.png', bbox_inches='tight')