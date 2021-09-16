#!/usr/bin/env python3
""" Transformer """
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    max_seq_len: integer representing the maximum sequence length
    dm: integer representing the model depth
    Returns: numpy.ndarray of shape (max_seq_len, dm) containing the positional
             encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    for row in range(max_seq_len):
        for col in range(0, dm, 2):
            PE[row, col] = np.sin(row / (10000 ** (col / dm)))
            PE[row, col + 1] = np.cos(row / (10000 ** (col / dm)))
    return PE


def sdp_attention(Q, K, V, mask=None):
    """
    Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
    K: tensor with shape (..., seq_len_v, dk) containing the key matrix
    V: tensor with shape (..., seq_len_v, dv) containing the value matrix
    mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
          containing the optional maask, or defaulted to None
    The Preceding dimensions of Q, K, and V are the same
    Returns: output, weights
             output: tensor with shape (..., seq_len_q, dv) containing the dot
                     product attention
             weights: tensor with shape (..., seq_len_q, seq_len_v) containing
                      the attention weights
    """
    # Matmul Q and K
    QK = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot product
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = QK / tf.math.sqrt(dk)

    # Add mask if not None
    if mask is not None:
        scaled += mask * -1e9

    # Pass scaled attention through softmax activation
    weights = tf.nn.softmax(scaled, axis=-1)

    # Matmul by value matrix for output
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi head attention
    """
    def __init__(self, dm, h):
        """
        dm: integer representing the model dimensionality
        h: integer representing the number of heads
        dm is divisible by h
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // self.h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension of tensor x into (h, depth)
        Transpose the result such that the shape is
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        return x

    def call(self, Q, K, V, mask):
        """
        Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
        K: tensor with shape (..., seq_len_v, dk) containing the key matrix
        V: tensor with shape (..., seq_len_v, dv) containing the value matrix
        mask: always None
        The Preceding dimensions of Q, K, and V are the same
        Returns: output, weights
                 output: tensor with shape (..., seq_len_q, dv) containing the
                         dot product attention
                 weights: tensor with shape (..., seq_len_q, seq_len_v)
                          containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Generate query, key, and value matrices
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split between heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot Product Attention
        attention, weights = sdp_attention(Q, K, V, mask)

        # Refit to pass through linear layer
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.dm))
        output = self.linear(attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class representation of an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        drop_rate: Dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x: tensor of shape (batch, input_seq_len, dm) containing the input to
           the encoder block
        training: boolean to determine if the model is training
        mask: the mask to be applied for multi head attention
        Returns: tensor of the shape (batch, input_seq_len, dm) containing the
                 block's output
        """
        # Pass through multi head attention and dropout layers
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        # Add and normalize
        out1 = self.layernorm1(x + attn_output)

        # Feed through dense layers and dropout layer
        dense_output = self.dense_hidden(out1)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout2(dense_output, training=training)

        # Add and normalize
        out2 = self.layernorm2(out1 + dense_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    """
    Class representation of an encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, max_seq_len,
                 drop_rate=0.1):
        """
        N: Number of blocks in the encoder
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        input_vocab: Size of the input vocabulary
        max_seq_len: Maximum sequence length possible
        drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.project = tf.keras.layers.Dense(dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for n in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.dense = tf.keras.layers.Dense(1250)
        self.dense_out = tf.keras.layers.Dense(5)

    def call(self, x, training, mask, batch_size):
        """
        x: tensor of shape (batch, input_seq_len, dm) containing the input to
           the encoder
        training: boolean to determine if the model is training
        mask: mask to the applied for multi head attention
        Returns: tensor of shape (batch, input_seq_len, dm) containing the
                 encoder output
        """
        # Project feature dimension to match model dimension
        x = self.project(x)

        print("=\nPass 1\n=")

        # Add positional encoding, pass through dropout layer
        x *= tf.math.sqrt(tf.cast(self.dm, 'float32'))
        x += self.positional_encoding
        x = self.dropout(x, training=training)

        print("=\nPass 2\n=")

        # Pass through each encoding block
        for block in self.blocks:
            x = block(x, training, mask)

        print("=\nPass 3\n=")
        x = self.dense(x)
        x = self.dense_out(x)

        return x
