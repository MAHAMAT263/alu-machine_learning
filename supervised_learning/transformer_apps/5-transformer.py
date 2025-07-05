#!/usr/bin/env python3
import tensorflow as tf


def scaled_dot_product_attention(Q, K, V, mask):
    """Calculate the attention weights.

    Args:
        Q: query shape (..., seq_len_q, depth)
        K: key shape (..., seq_len_k, depth)
        V: value shape (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dm, h):
        super().__init__()
        assert dm % h == 0, "dm must be divisible by h"
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth).
        Transpose for shape: (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)  # (batch_size, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, h, seq_len_k, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, h, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, h, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))  # (batch_size, seq_len_q, dm)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, dm)
        return output, attention_weights


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, dm, hidden):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(dm)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.ffn = PositionWiseFeedForward(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual + LayerNorm

        ffn_output = self.ffn(out1)  # Feed-forward
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual + LayerNorm

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = PositionWiseFeedForward(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # Masked multi-head attention (self-attention)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Multi-head attention with encoder output
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)

        self.enc_layers = [EncoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Add embedding and position encoding
        x = self.embedding(x)  # (batch_size, seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, seq_len, dm)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)

        self.dec_layers = [DecoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights  # (batch_size, seq_len, dm), attention weights dict


class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_len, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_len, dropout_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_len, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, targets, training, enc_mask, look_ahead_mask, dec_mask):
        enc_output = self.encoder(inputs, training, enc_mask)
        dec_output, attention_weights = self.decoder(targets, enc_output, training, look_ahead_mask, dec_mask)
        final_output = self.final_layer(dec_output)  # logits
        return final_output, attention_weights


def positional_encoding(seq_len, dm):
    """Create positional encoding for sequences"""
    angles = get_angles(tf.range(seq_len)[:, tf.newaxis],
                        tf.range(dm)[tf.newaxis, :],
                        dm)
    angles[:, 0::2] = tf.math.sin(angles[:, 0::2])  # even indices
    angles[:, 1::2] = tf.math.cos(angles[:, 1::2])  # odd indices
    pos_encoding = angles[tf.newaxis, ...]  # (1, seq_len, dm)
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, dm):
    """Helper for positional encoding"""
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(dm, tf.float32))
    return tf.cast(pos, tf.float32) * angle_rates
