#!/usr/bin/env python3
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation in the transformer.

    Args:
        inputs: tf.Tensor, shape (batch_size, seq_len_in), input sentence tokens
        target: tf.Tensor, shape (batch_size, seq_len_out), target sentence tokens

    Returns:
        encoder_mask: padding mask for encoder (batch_size, 1, 1, seq_len_in)
        combined_mask: look-ahead + padding mask for decoder first attention block (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: padding mask for decoder second attention block (batch_size, 1, 1, seq_len_in)
    """

    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)  # shape (batch_size, seq_len_in)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]    # (batch_size, 1, 1, seq_len_in)

    # Decoder padding mask for 2nd attention block
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)  # same as encoder padding mask
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]     # (batch_size, 1, 1, seq_len_in)

    # Look-ahead mask for target (prevents attending to future tokens)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # shape (seq_len, seq_len), upper triangular matrix with 1s above diagonal

    # Target padding mask
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)  # (batch_size, seq_len_out)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len_out)
    # We want to combine look ahead mask and padding mask to apply to the target sequence

    # Combine look ahead and padding masks:
    # Broadcast look_ahead_mask to (1, 1, seq_len_out, seq_len_out)
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
