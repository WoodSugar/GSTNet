# -*- coding: utf-8 -*-
"""
@Time   : 2019/02/28 15:54

@Author : Yuppie
"""
import os
import csv
import pickle
import argparse
import numpy as np
import pandas as pd


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape

    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args, x_offset, y_offset):
    df = pd.read_hdf(args.traffic_filename)
    x, y = generate_graph_seq2seq_io_data(df=df,
                                          x_offsets=x_offset,
                                          y_offsets=y_offset,
                                          add_time_in_day=True,
                                          add_day_in_week=False)
    x = x.transpose((0, 2, 1, 3))
    y = y.transpose((0, 2, 1, 3))
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = round(x.shape[0] / 8)
    num_train = round(num_samples * 0.7)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[num_train+num_val: num_train+num_val+num_test], y[num_train+num_val: num_train+num_val+num_test]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "PeMS_Bay_%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offset.reshape(list(x_offset.shape) + [1]),
            y_offsets=y_offset.reshape(list(y_offset.shape) + [1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="DCRNN_Provide/", help="Output directory.")
    parser.add_argument("--traffic_filename", type=str, default="DCRNN_Provide/pems-bay.h5",
                        help="Raw traffic readings.")

    args = parser.parse_args()

    generate_train_val_test(args, np.sort(np.concatenate((np.arange(-4*3*24, 0, 3),))), np.sort(np.arange(0, 1, 1)))

    # sensor_id, id2ind, adj = load_graph_data("DCRNN_Provide/adj_mx_bay.pkl")
    #
    # print(type(adj))
    # with open("pems_sensor_id.csv", "w", newline="") as fw:
    #     writer = csv.writer(fw)
    #     for i in range(len(sensor_id)):
    #         writer.writerow([int(sensor_id[i])])
    # fw.close()
