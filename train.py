#!/usr/bin/env python3
""" creates and trains a model """
# leaving out ependencies for now
# import dataset f'n
# import create masks f'n
# import actual transformer file
# import hyperparameters as HP


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ createa and trains a transformer model for finishing sketches
        N: number of encoder blocks
        dm: dimensionality of model
        h: number of heads
        hidden: number of hidden units in fc layer
        max_len: max tokens per seq
        batch_size: batch size for training
        epochs: no. of epochs to train for

        Return: the trained model
    """
    # get the dataset and break into batches
    ds = Dataset(file_path, batch_size, max_len)

    # create the transformer architecture
    transformer = Encoder(6, 1024, 8, 2048, 250)

    # create loss object to use in custom loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # create masks function for training
    def create_mask(mask_rate, length):
        mask = tf.random.uniform(shape=(length,), minval=0, maxval=1)
        mask = tf.map_fn(lambda x: 0 if x < mask_rate else 1, mask)
        mask = tf.cast(mask, 'float64')
        return mask[:, tf.newaxis]

    # create the actucal loss function
    def loss_function(real, pred, mask=None):
        """ custom loss function for our transformer
            real: the real values of the output
            pred: the predicted vlaues from the model
            mask: mask to block off parts of real and pred
        """
        if mask:
            masked_real = mask * real
            masked_pred = mask * pred
            loss_ = loss_object(masked_real, masked_pred)
            return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
        else:
            loss = loss_object(real, pred)
            return tf.reduce_sum(loss_)

    # initialize some hyper param choices we'll need later
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    # train_step function
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        """ single train step 
            takes in input and a target value
        """
        # create mask(hide_rate)
        train_mask = create_mask(rate, batch_size, length)

        # crate the target input vs real this is where we'll do our masking
        tar_real = tar
        tar_masked = tar * train_mask

        # then create masks, at least 2/3 now not needed because we went encoder only
        # enc_p_mask, comb_mask, dec_p_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            # mask None because i believe it's just the padd mask
            predictions = transformer(inp,
                                      training=True,
                                      mask=None,
                                      batch_size=batch_size)
            # i put in third arg but not yet totally sure
            loss = loss_function(tar_real, predictions, mask=train_mask)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
             zip(gradients, transformer.trainable_variables))
        train_loss(loss)

    # process through epochs training
    for epoch in range(epochs):
        train_loss.reset_states()
        for (batch, (inp, tar)) in enumerate(ds.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch}' +
                      ' Loss {train_loss.result():.4f}' +
                      ' Accuracy {train_accuracy.result():.4f}')
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}' +
              ' Accuracy {train_accuracy.result():.4f}')
    return transformer
