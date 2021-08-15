from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from scipy import sparse

"""
used to generate dataset. There are three pairs of dataset in total.
1. train dataset (contain  2018-06 to  2018-07 yellow taxi):  train_X.npz / train_y.npy
2. validation (contain  2018-08 yellow taxi): valid_X.npz / valid_y.npy
3. test dataset (contain  2018-09 yellow taxi): test_X.npz / test_y.npy
"""

columns_drop = ['VendorID', 'store_and_fwd_flag', 'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                'tpep_pickup_datetime', 'tpep_dropoff_datetime']
columns_continuous = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']
columns_discrete = ['PULocationID', 'DOLocationID', 'payment_type', 'weekday', 'hour', 'weather']


def split_x_y(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    split predictor and response
    :param df: whole dataset
    :return:
        y: response
    """
    y = df['trip_time_cost']
    return y


def adjust_time_type(df: pd.DataFrame):
    """
    change time from str to time and add trip_time_cost (units is min)
    :param df: raw data frame
    :return: change time from str to time and add trip_time_cost (units is min)
    """
    df.dropna(inplace=True)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['trip_time_cost'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds.astype(float) / 60
    # inplace modify, no need to return


def del_extreme_value(df: pd.DataFrame, start_time, end_time) -> pd.DataFrame:
    """
    delete extreme value
    :param end_time: data frame pickup_datetime lower bound
    :param start_time: data frame pickup_datetime upper bound
    :param df:
    """
    start_y, start_m, start_d = start_time
    end_y, end_m, end_d = end_time

    df = df.loc[
        (df['tpep_pickup_datetime'] > pd.Timestamp(start_y, start_m, start_d)) &
        (df['tpep_pickup_datetime'] < pd.Timestamp(end_y, end_m, end_d)) &
        (df["trip_distance"] > 0) & (df["trip_distance"] <= 60) &
        (df["tip_amount"] > 0) & (df["tip_amount"] <= 50) &
        (df["fare_amount"] > 0) & (df["fare_amount"] <= 200) &
        (df["trip_time_cost"] > 0) & (df["trip_time_cost"] <= 200) &
        (df["passenger_count"] > 0) & (df["passenger_count"] <= 4) &
        (df["RatecodeID"] == 1)
        ]
    df.reset_index(drop=True, inplace=True)
    return df
    # inplace modify, no need to return


def generate_new_attr(df: pd.DataFrame, weather: pd.DataFrame):
    """
    generate weekend/hour/peak attribute based on pickup time
    """
    df['weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    # only used to merge weather
    df['time_index'] = df['tpep_pickup_datetime'].dt.date.astype(str) + '-' + df['hour'].astype(str)
    df['weather'] = df['time_index'].apply(lambda x: weather[x])

    # inplace modify, no need to return


def onehot_minmax_adjust(df: pd.DataFrame, onehoe_enc, scaler) -> np.array:
    """
    for discrete attribute, do one hot operation. for continuous attribute, do min max scale.
    """
    # Scaling continuous attributes

    if onehoe_enc is None:
        onehoe_enc = OneHotEncoder(handle_unknown='ignore')
        onehoe_enc.fit(df[columns_discrete])
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[columns_continuous])

    continuous_part = scaler.transform(df[columns_continuous])
    # One hot discrete attributes
    discrete_part = onehoe_enc.transform(df[columns_discrete])
    # return np.concatenate((continuous_part, discrete_part), axis=1)
    return sparse.hstack((sparse.csr_matrix(continuous_part), discrete_part)), onehoe_enc, scaler


def preprocessing(df: pd.DataFrame, weather: pd.DataFrame, start_time, end_time, onehoe_enc=None, scaler=None):

    file_name = '_'.join(str(i) for i in start_time + end_time)

    # preprocess weather file
    weather = weather.stack().reset_index()
    weather['time'] = pd.to_datetime(weather['time']).astype(str) + '-' + weather['level_1'].astype(str)
    weather.set_index(['time'], inplace=True)
    weather = weather[0]

    adjust_time_type(df)
    df = del_extreme_value(df, start_time, end_time)
    df_y = split_x_y(df)
    generate_new_attr(df, weather)

    df.to_feather('../preprocessed_data/' + file_name + '.feather')

    df_X, onehoe_enc, scaler = onehot_minmax_adjust(df, onehoe_enc, scaler)
    return df_X, df_y, onehoe_enc, scaler


if __name__ == '__main__':
    weather = pd.read_csv('../raw_data/weather.csv', index_col=0)

    train_df = pd.concat([pd.read_csv('../raw_data/yellow_tripdata_2018-06.csv'),
                          pd.read_csv('../raw_data/yellow_tripdata_2018-07.csv')])
    train_X, train_y, onehoe_enc, scaler = preprocessing(train_df, weather, (2018, 6, 1), (2018, 7, 31))
    sparse.save_npz('../preprocessed_data/train_X.npz', train_X)
    np.save('../preprocessed_data/train_y.npy', train_y)

    valid_df = pd.read_csv('../raw_data/yellow_tripdata_2018-08.csv')
    valid_X, valid_y, _, _ = preprocessing(valid_df, weather, (2018, 8, 1), (2018, 8, 31), onehoe_enc, scaler)
    sparse.save_npz('../preprocessed_data/valid_X.npz', valid_X)
    np.save('../preprocessed_data/valid_y.npy', valid_y)

    test_df = pd.read_csv('../raw_data/yellow_tripdata_2018-09.csv')
    test_X, test_y, _, _ = preprocessing(test_df, weather, (2018, 9, 1), (2018, 9, 30), onehoe_enc, scaler)
    sparse.save_npz('../preprocessed_data/test_X.npz', test_X)
    np.save('../preprocessed_data/test_y.npy', test_y)
