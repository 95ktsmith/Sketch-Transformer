#!/usr/bin/env python3

"""
Utility class for handling the loading, pre-processing, and post-processing
of data
"""
import numpy as np
import utils


# Tokenizing isn't being used anymore but these functions are being kept
# until we're absolutely certain we don't need them

def strokes_to_tokens(strokes):
    """
    strokes is a 2d numpy array of stroke-5 vectors to be converted into tokens
    Returns a list of the tokenized vectors
    """
    tokens = []
    for stroke in strokes:
        token = (stroke[0] + 255) * 511
        token += (stroke[1] + 255)
        token += np.sum(np.array([100, 500000, 1000000]) * stroke[2:])
        tokens.append(token)
    return np.asarray(tokens)

def tokens_to_strokes(tokens):
    """
    tokens is a 1d numpy array of stroke tokens to be converted into strokes
    Returns a list of the strokes
    """
    strokes = []
    for token in tokens:
        stroke = [0] * 5
        if token // 1000000 == 1:
            stroke[4] = 1
            token -= 1000000
        elif token // 500000 == 1:
            stroke[3] = 1
            token -= 500000
        else:
            stroke[2] = 1
            token -= 100
        stroke[0] = token // 511 - 255
        stroke[1] = token % 511 - 255
        strokes.append(stroke)
    return np.asarray(strokes)

def clean(data, max_length=250):
    """
    Data is a np 3d array of samples in stroke-3 format
    Removes all samples with length > max_length
    Converts to stroke-5 and pads to max_length
    Tokenizes stroke-5 vectors
    Returns tokenized dataset as a np 2d array
    """
    dataset = []
    for sample in data:
        if len(sample) <= max_length:
            sample = utils.to_big_strokes(sample, max_length)
            dataset.append(sample)
    return np.asarray(dataset)

class Dataset:
    """ Document later """

    def __init__(self, filepath, max_length=250):
        """ Init """
        data = np.load(
            filepath,
            encoding='latin1',
            allow_pickle=True
        )

        # Clean up dataset, removing samples over max_length
        # and tokenizing
        self.train = clean(data['train'])
        self.valid = clean(data['valid'])
        self.test = clean(data['test'])
