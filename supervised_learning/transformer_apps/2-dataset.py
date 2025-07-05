#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    '''Loads and prepares a dataset for machine translation'''

    def __init__(self):
        '''Class constructor'''
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Apply encoding to the datasets
        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
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
        '''
        Encodes Portuguese and English sentences into subword tokens
        Includes start and end tokens
        '''
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
                    self.tokenizer_pt.encode(pt.numpy()) + \
                    [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + \
                    self.tokenizer_en.encode(en.numpy()) + \
                    [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        '''
        TensorFlow wrapper for the encode method
        Ensures compatibility with tf.data.Dataset map()
        '''
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
