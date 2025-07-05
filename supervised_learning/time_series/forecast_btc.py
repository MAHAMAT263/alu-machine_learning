#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def load_data():
    data = np.load('btc_data.npz')
    X = data['X']
    y = data['y']
    return X, y

def create_datasets(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(input_shape):
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    X, y = load_data()
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_dataset = create_datasets(X_train, y_train)
    val_dataset = create_datasets(X_val, y_val)

    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    print("✅ Model training complete.")

if __name__ == '__main__':
    main()
