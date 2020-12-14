# -*- coding: utf-8 -*-
"""
@Time   : 2019/02/28 23:01

@Author : Yuppie
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from Process_DCRNN_Data import load_graph_data


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, keywords):
    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, 'PeMS_Bay_' + category + '.npz'))
        data['x_' + category] = cat_data['x'][:, :, :, 0][:, :, :, np.newaxis]
        data['y_' + category] = cat_data['y'][:, :, :, 0][:, :, :, np.newaxis]

    station_mean = np.mean(data['x_train'], axis=(0, 2))[np.newaxis, :, np.newaxis]
    station_std = np.std(data['x_train'], axis=(0, 2))[np.newaxis, :, np.newaxis]
    scaler = StandardScaler(mean=station_mean, std=station_std)

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

    if keywords == 'train':
        return data['x_train'], data['y_train'], scaler
    elif keywords == 'val':
        return data['x_val'], data['y_val'], scaler
    elif keywords == 'test':
        return data['x_test'], data['y_test'], scaler


class ToTensor(object):
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, input_pair):
        x = input_pair[0]
        y = input_pair[1]
        tensor_x = torch.tensor(x, dtype=self.dtype, device=self.device)
        tensor_y = torch.tensor(y, dtype=self.dtype, device=self.device)
        return tensor_x, tensor_y


class Create_Dataset(Dataset):
    def __init__(self, dataset_dir, mode, dtype, device):
        self.data_x, self.data_y, self.scaler = load_dataset(dataset_dir, mode)

        self.transform = ToTensor(dtype, device)
        self.length = self.data_x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.transform([self.data_x[index], self.data_y[index]])


class Load_Graph(object):
    def __init__(self, pkl_filename):
        self.filename = pkl_filename

    def load_graph(self, K):
        _, _, base_graph = load_graph_data(self.filename)

        graph = base_graph
        for _ in range(base_graph-1):
            graph = np.dot(graph, base_graph)

        graph = torch.from_numpy(np.array(graph))
        graph = torch.gt(graph, 0)

        graph = graph.to(dtype=torch.float)

        return graph


if __name__ == '__main__':
    dataset = Create_Dataset("DCRNN_Provide", "train", torch.float, torch.device("cpu"))
    print(dataset[0][0].size())
    print(dataset[0][1].size())
