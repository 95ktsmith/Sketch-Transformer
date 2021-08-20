#!/usr/bin/env python3

import numpy as np
strokes_to_tokens = __import__('tokenizer').strokes_to_tokens
tokens_to_strokes = __import__('tokenizer').tokens_to_strokes

print("Sample strokes, before tokenizing")
strokes = np.asarray([
    [11, 40, 0, 1, 0],
    [80, 150, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [-255, -255, 1, 0, 0],
    [255, 255, 0, 0, 1]
])
print(strokes)

print("\nTokenized strokes")
tokens = strokes_to_tokens(strokes)
print(tokens)

print("\nTokens convered back to strokes")
t_strokes = tokens_to_strokes(tokens)
print(t_strokes)

print("\nNormalized tokens")
print(tokens / 783362)
