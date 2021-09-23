#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Function from utils.py for sketch-rnn in the Magenta github repository
# at https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn
def to_big_strokes(stroke, max_len=100):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result

def clean(data, max_length=100):
    """
    Data is a np 3d array of samples in stroke-3 format
    Removes all samples with length > max_length
    Converts to stroke-5 and pads to max_length
    """
    dataset = []
    for sample in data:
        if len(sample) <= max_length:
            sample = to_big_strokes(sample, max_length)
            dataset.append(sample)
    dataset = np.asarray(dataset)
    return dataset


class Dataset:
    """ Loads a numpy.npz file to be used for training """

    def __init__(self,
                 filepath,          # Path to file to load 
                 batch_size=32,     # Batch size to use
                 max_length=250):   # Maximum sequence length per example

        data = np.load(
            filepath,
            encoding='latin1',
            allow_pickle=True
        )

        # Clean up dataset, removing samples over max_length
        self.train = clean(data['train'], max_length)
        self.valid = clean(data['valid'], max_length)
        self.test = clean(data['test'], max_length)

        # Convert to tensorflow datasets for training
        self.train = tf.convert_to_tensor(self.train)
        self.train = tf.data.Dataset.from_tensor_slices(list(self.train))
        self.valid = tf.convert_to_tensor(self.valid)
        self.valid = tf.data.Dataset.from_tensor_slices(list(self.valid))
        self.test = tf.convert_to_tensor(self.test)
        self.test = tf.data.Dataset.from_tensor_slices(list(self.test))

        # Shuffle and batch train and valid sets
        self.train = self.train.shuffle(max_length)
        self.valid = self.valid.shuffle(max_length)
        self.train = self.train.batch(batch_size)
        self.valid = self.valid.batch(batch_size)
