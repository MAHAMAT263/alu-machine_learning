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

    def tokenize_dataset(self, data):
        '''
        Creates sub-word tokenizers for the dataset
        Args:
            data: tf.data.Dataset with (pt, en) sentence pairs
        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        '''
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en
