#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    '''Loads and prepares a dataset for machine translation'''

    def __init__(self, batch_size, max_len):
        '''Class constructor with batch_size and max_len'''
        self.batch_size = batch_size
        self.max_len = max_len

        # Load raw datasets
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Build tokenizers from training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Encode the datasets (map the encoding)
        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Filter out long sentences (both Portuguese and English)
        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_valid = self.data_valid.filter(self.filter_max_length)

        # Cache the training dataset for performance
        self.data_train = self.data_train.cache()

        # Shuffle the training dataset with a buffer size of dataset size (or a large number)
        self.data_train = self.data_train.shuffle(buffer_size=10000)

        # Batch and pad the training dataset
        self.data_train = self.data_train.padded_batch(
            batch_size,
            padded_shapes=(
                [None],  # pt sentences padded to max length in batch
                [None]   # en sentences padded to max length in batch
            )
        )

        # Prefetch for better performance
        self.data_train = self.data_train.prefetch(tf.data.AUTOTUNE)

        # Batch and pad the validation dataset
        self.data_valid = self.data_valid.padded_batch(
            batch_size,
            padded_shapes=(
                [None],  # pt sentences padded to max length in batch
                [None]   # en sentences padded to max length in batch
            )
        )

    def tokenize_dataset(self, data):
        '''Creates sub-word tokenizers for the dataset'''
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''Encode Portuguese and English sentences into tokens with start/end'''
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
                    self.tokenizer_pt.encode(pt.numpy()) + \
                    [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + \
                    self.tokenizer_en.encode(en.numpy()) + \
                    [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        '''TensorFlow wrapper for encode() to use in dataset map'''
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def filter_max_length(self, pt, en):
        '''Filter function to keep only sentences <= max_len tokens'''
        return tf.logical_and(tf.size(pt) <= self.max_len,
                              tf.size(en) <= self.max_len)
