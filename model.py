#!/usr/bin/env python3

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


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class representation of a decoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1, name=None):
        """
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        drop_rate: Dropout rate
        """
        super(DecoderBlock, self).__init__()
        if name is not None:
            self._name = name
        self.mha1 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, look_ahead_mask, training=False):
        """
        x: tensor of shape (batch, target_seq_len, dm)containing the input to
           the decoder block
        training: boolean to determine if the model is training
        look_ahead_mask: mask to be applied to the first multi head attention
                         layer
        Returns: tensor of shape (batch, target_seq_len, dm) containing the
                 block's output
        """
        # Pass through MHA and dropout layer
        attn_out, _ = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        attn_out = self.dropout1(attn_out, training=training)

        # Add and normalize
        out = self.layernorm1(inputs + attn_out)

        # Pass through dense layers and dropout layer
        dense_output = self.dense_hidden(out)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout2(dense_output, training=training)

        # Add and normalize
        out = self.layernorm2(out + dense_output)

        return out


class BranchedDecoder(tf.keras.Model):
    """
    A branched Transformer Decoder

    The model uses a linear layer to bring the input feature dimension up to
    the model's dimension before adding fixed sinusoidal positional encoding.

    The model passes the encoded inputs through (Nb) decoder blocks before
    passing their final output to two branches, of (No) and (Np) decoder blocks
    respectively, that predict the X and Y offsets, and pen state
    probabilities, of the next point for each given point in the sequence.

    The offset branch produces an output of size (batch, sequence_len, 2),
    where the feature dimension is [X offset, Y Offset] from the previous point.

    The pen state branch produces an output of size (batch, sequence_len, 3),
    where the feature dimension is [p0, p1, p2], each representing the
    probabilities that the pen will be down, up, or finished, respectively.

    Returns: offsets_predictions, pen_state_predictions
    """
    def __init__(self,
                 Nb,                # Number of blocks in model base
                 No,                # Number of blocks in offset branch
                 Np,                # Number of blocks in pen state branch
                 dm,                # Model dimensionality
                 h,                 # Number of heads used in attention
                 hidden,            # Hidden layer dimenssionality
                 max_seq_len,       # Maximum sequence length
                 drop_rate=0.1):    # Drop rate used in dropout layers

        super(BranchedDecoder, self).__init__()
        self.Nb = Nb
        self.No = No
        self.Np = Np
        self.dm = dm
        self.projection = tf.keras.layers.Dense(dm, name='base_projection')
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

        self.base_blocks = [
            DecoderBlock(dm, h, hidden, drop_rate,
            name="base_block_" + str(n)) for n in range(Nb)
        ]
        self.offset_blocks = [
            DecoderBlock(dm, h, hidden, drop_rate,
            name="offset_block_" + str(n)) for n in range(No)
        ]
        self.pen_blocks = [
            DecoderBlock(dm, h, hidden, drop_rate,
            name="pen_block_" + str(n)) for n in range(Np)
        ]

        self.offset_dense = tf.keras.layers.Dense(dm, name='offset_dense')
        self.offset_out = tf.keras.layers.Dense(2, name='offset_out')
        self.pen_dense = tf.keras.layers.Dense(dm, name='pen_dense')
        self.pen_out = tf.keras.layers.Dense(3, name='pen_out',
                                             activation='softmax')

    def call(self,
             inputs,                # Input data
             look_ahead_mask=None,  # Mask used for attention
             training=False):       # Whether the model is training or not

        seq_len = int(inputs.shape[1])

        # Project to model dimension
        x = self.projection(inputs)

        # Add positional encoding and pass through dropout layer
        x *= tf.math.sqrt(tf.cast(self.dm, 'float32'))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        # Pass through base decoder blocks
        for block in self.base_blocks:
            x = block(x, look_ahead_mask, training)

        # Pass base output through offset branch
        offset = x
        for block in self.offset_blocks:
            offset = block(offset, look_ahead_mask, training)
        offset = self.offset_dense(offset)
        offset = self.offset_out(offset)

        # Pass base output through pen state branch
        pen = x
        for block in self.pen_blocks:
            pen = block(pen, look_ahead_mask, training)
        pen = self.pen_dense(pen)
        pen = self.pen_out(pen)

        return offset, pen
