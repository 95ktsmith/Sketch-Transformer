#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from model import BranchedDecoder
from dataset import Dataset

def train_model(Nb,           # Number of blocks in model base
                No,           # Number of blocks in offset branch
                Np,           # Number of blocks in pen state branch
                dm,           # Model dimensionality
                h,            # Number of heads used in attention
                hidden,       # Hidden layer dimensionality
                max_len,      # Maximum sequence length
                batch_size,   # Batch size
                epochs,       # Number of epochs to train for
                filepath,     # Path to file to use for training dataset
                verbose=1,    # 0: No printing, 1: Print loss after each epoch,
                              # 2: Print loss every 50 epochs
                weights=None):# Path to weights to use for continuing training
                              # If none, model weights will be initialized
    """
    Creates and trains a model used for predicting future points in an
    unfinished drawing from Google's Quick, Draw! dataset.

    The offset prediction branch is trained using Mean Squared Error loss, and
    the pen state prediction branch is trained using Categorical Crossentropy
    loss.

    The model's weights are saved and overwritten after each epoch if they are
    the best performing at the time.

    Returns: The model, MSE loss history, CCE loss history
    """

    # Load dataset
    data = Dataset(filepath, batch_size=batch_size, max_length=max_len)
    
    # Create model
    model = BranchedDecoder(Nb, No, Np, dm, h, hidden, max_len)

    # Run a dummy set of inputs through to initialize weights
    inputs = np.random.uniform(size=(1, max_len, 5))
    model(inputs, None)

    # Load weights if continuing training
    if weights is not None:
        model.load_weights(weights)

    # Create lists of weights to apply gradients to.
    # Done to separate loss and gradients between offset and pen state 
    # branches, while still applying both to the shared base
    offset_weights = []
    pen_weights = []
    for weight in model.trainable_weights:
        if "base" in weight.name:
            offset_weights.append(weight)
            pen_weights.append(weight)
        if "offset" in weight.name:
            offset_weights.append(weight)
        if "pen" in weight.name:
            pen_weights.append(weight)

    # Loss functions, metrics, learning rate scheduler, optimizers
    offset_loss_func = tf.keras.losses.MeanSquaredError()
    pen_loss_func = tf.keras.losses.CategoricalCrossentropy()

    pen_train_loss = tf.keras.metrics.Mean(name='pen_train_loss')
    offset_train_loss = tf.keras.metrics.Mean(name='offset_train_loss')

    learning_rate = 0.0001

    offset_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    pen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Sequence length is constant throughout the dataset, so the attention
    # mask can be made ahead of time
    mask = 1 - tf.linalg.band_part(tf.ones((max_len - 1, max_len - 1)), -1, 0)

    # Create lists to store loss histories
    offset_losses = []
    pen_losses = []

    # High arbitrary number to begin comparing best loss to
    prev_best = 10000

    # Define training step
    def train_step(inputs, real):
        """ Single training step """

        # Create gradient tape and get predictions from the model
        with tf.GradientTape(persistent=True) as tape:
            offsets, pen_states = model(inputs, mask, True)

            # Calculate losses
            offset_loss = offset_loss_func(real[:, :, :2], offsets)
            pen_loss = pen_loss_func(real[:, :, 2:], pen_states)

        # Apply gradients to offset branch & base
        grads = tape.gradient(offset_loss, offset_weights)
        offset_optimizer.apply_gradients(zip(grads, offset_weights))

        # Apply gradients to pen state branch & base
        grads = tape.gradient(pen_loss, pen_weights)
        pen_optimizer.apply_gradients(zip(grads, pen_weights))

        # Update loss states
        offset_train_loss(offset_loss)
        pen_train_loss(pen_loss)

        del tape

    # Training Loop
    for epoch in range(epochs):

        # Reset loss metrics at the start of the epoch
        offset_train_loss.reset_states()
        pen_train_loss.reset_states()

        for batch, inp in enumerate(data.train):

            # Target values are input values shifted right by one step
            train_step(inp[:, :-1], inp[:, 1:])

            # Update loss histories
            offset_losses.append(offset_train_loss.result())
            pen_losses.append(pen_train_loss.result())

            if verbose == 2:  # Print results every 50 batches
                if batch % 50 == 0:
                    if batch % 50 == 0:
                        print("Epoch {}, batch {}: Offset Loss: {} Pen Loss {}"
                        .format(
                            epoch + 1,
                            batch,
                            offset_train_loss.result(),
                            pen_train_loss.result()
                        ))

        if verbose >= 1:  # Print results after each epoch
            print("Epoch {}: Offset Loss: {:.4f} Pen Loss {:.4f}".format(
                epoch + 1,
                offset_train_loss.result(),
                pen_train_loss.result()
            ))

        # Save best performing weights
        if offset_train_loss.result() < prev_best:
            model.save_weights('{}_epoch_best.h5'.format(epochs))
            prev_best = offset_train_loss.result()

    return model, offset_losses, pen_losses


if __name__ == "__main__":
    model, offset_losses, pen_losses = train_model(
    4,              # Base blocks
    2,              # Offset blocks
    2,              # Pen state blocks
    128,            # Model dimensionality
    8,              # Heads
    512,            # Hidden units
    100,            # Max sequence length
    64,             # Batch size
    1,              # Epochs
    'data/cat.npz', # File path
    2,              # Verbosity
    None)           # Weights file if continuing training