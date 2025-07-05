#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(y_true, y_pred):
    """
    Sparse categorical crossentropy loss, ignoring padding (0 tokens).
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss_ = loss_object(y_true, y_pred) * mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(y_true, y_pred):
    """
    Sparse categorical accuracy, ignoring padding tokens (0).
    """
    y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.cast(tf.equal(y_true, y_pred_ids), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Create and train the transformer model on Portuguese to English translation.

    Args:
        N: number of encoder/decoder blocks
        dm: model dimensionality
        h: number of heads
        hidden: hidden units in feedforward layers
        max_len: max tokens per sequence
        batch_size: training batch size
        epochs: number of epochs

    Returns:
        trained Transformer model
    """

    dataset = Dataset(batch_size, max_len)

    # Instantiate model
    model = Transformer(
        N=N, dm=dm, h=h, hidden=hidden,
        input_vocab=dataset.tokenizer_pt.vocab_size + 2,
        target_vocab=dataset.tokenizer_en.vocab_size + 2,
        max_seq_len=max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    # Prepare metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp,
                                   True,
                                   enc_mask,
                                   combined_mask,
                                   dec_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        acc = accuracy_function(tar_real, predictions)

        train_loss(loss)
        train_accuracy(acc)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch_num, (inp, tar) in enumerate(dataset.data_train):
            train_step(inp, tar)

            if batch_num % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {batch_num}: loss {train_loss.result():.6f} accuracy {train_accuracy.result():.6f}")

        print(f"Epoch {epoch + 1}: loss {train_loss.result():.6f} accuracy {train_accuracy.result():.6f}")

    return model
