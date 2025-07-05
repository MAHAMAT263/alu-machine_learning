#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_merge_data():
    # Load data
    coinbase = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
    bitstamp = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

    # Convert timestamp to datetime
    coinbase['Timestamp'] = pd.to_datetime(coinbase['Timestamp'], unit='s')
    bitstamp['Timestamp'] = pd.to_datetime(bitstamp['Timestamp'], unit='s')

    # Merge on timestamp
    df = pd.merge(coinbase, bitstamp, on='Timestamp', suffixes=('_coinbase', '_bitstamp'))

    # Take average of closing prices
    df['Close'] = df[['Close_coinbase', 'Close_bitstamp']].mean(axis=1)

    # Keep timestamp and close price
    df = df[['Timestamp', 'Close']].dropna()

    return df


def resample_hourly(df):
    df.set_index('Timestamp', inplace=True)
    hourly_df = df.resample('1H').mean().dropna()
    return hourly_df


def create_sequences(data, history_steps=24, forecast_steps=1):
    X, y = [], []
    for i in range(len(data) - history_steps - forecast_steps + 1):
        X.append(data[i:i+history_steps])
        y.append(data[i+history_steps+forecast_steps-1])
    return np.array(X), np.array(y)


def main():
    df = load_and_merge_data()
    df = resample_hourly(df)

    # Normalize close prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    # Create sequences
    X, y = create_sequences(scaled, history_steps=24, forecast_steps=1)

    # Save
    np.savez('btc_data.npz', X=X, y=y, scaler_min=scaler.data_min_, scaler_max=scaler.data_max_)
    print("✅ Preprocessing complete. Data saved to btc_data.npz")


if __name__ == '__main__':
    main()
