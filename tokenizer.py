#!/usr/bin/env python3
"""
Tokenizer for converting Stroke-5 vectors to tokens and tokens to
Stroke-5 Vectors
"""

import numpy as np

def strokes_to_tokens(strokes):
    """
    strokes is a 2d numpy array of stroke-5 vectors to be converted into tokens
    Returns a list of the tokenized vectors
    """
    tokens = []
    for stroke in strokes:
        token = (stroke[0] + 255) * 1533
        token += (stroke[1] + 255) * 3
        token += np.argwhere(stroke[2:] == 1)[0, 0]
        tokens.append(token)
    return np.asarray(tokens)

def tokens_to_strokes(tokens):
    """
    tokens is a 1d numpy array of stroke tokens to be converted into strokes
    Returns a list of the strokes
    """
    strokes = []
    for token in tokens:
        stroke = []
        stroke.append(token // 1533 - 255)
        stroke.append((token % 1533) // 3 - 255)
        stroke += [0, 0, 0]
        stroke[((token % 1533) % 3) + 2] = 1
        strokes.append(stroke)
    return np.asarray(strokes)
