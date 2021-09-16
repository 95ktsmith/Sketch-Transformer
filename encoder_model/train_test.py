#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from transformer import Encoder
from utils import to_big_strokes
from dataset import Dataset

model = Encoder(6, 1024, 8, 2048, 250)

data = Dataset('data/cat.npz')

batch = data.train[0:50]

output = model(batch, False, None, 50)

# havent yet finalized arguments
trained = train_transformer(N, dm, h, hidden, max_len, batch_size, epochs)

print(trained)
