# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 17:26

@Author : Yuppie
"""
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from Train import train, test
from Model_Size import model_size
from Transform_DataSet import Create_Dataset
from Process_DCRNN_Data import load_graph_data
from Sublayers import Graph2S
from Models import Prediction_Model

from Compare_Methods.LSTM import LSTM_Model
from Compare_Methods.ChebNet import Merge_ChebNet
from Compare_Methods.GCGRU import Merge_GCGRU
from Compare_Methods.ST_GCN import STGCN_Model

from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_name", type=str, default="DCRNN_Provide")

    parser.add_argument("-beta", type=float, default=2)
    parser.add_argument("-num_of_graph", type=int, default=1)
    parser.add_argument("-num_of_neighbors", type=int, default=2)
    parser.add_argument("-num_of_stations", type=int, default=325)
    parser.add_argument("-temporal_in_channel", type=int, default=1)
    parser.add_argument("-spatial_out_channel", type=int, default=4)
    parser.add_argument("-switch", type=str, default="dot_product")

    parser.add_argument("-epoch", type=int, default=600)
    parser.add_argument("-batch_size", type=int, default=48)

    parser.add_argument("-log", default=None)
    parser.add_argument("-save_model", default=None)
    parser.add_argument("-save_mode", type=str, choices=["all", "best"], default="best")

    option = parser.parse_args()
    option.save_model = "Result_Models/STGCN"

    device = torch.device("cpu")

    training_dataset = DataLoader(Create_Dataset(option.file_name,
                                                 mode="train",
                                                 dtype=torch.float,
                                                 device=device),
                                  batch_size=option.batch_size, shuffle=True, num_workers=32)

    validation_dataset = DataLoader(Create_Dataset(option.file_name,
                                                   mode="val",
                                                   dtype=torch.float,
                                                   device=device),
                                    batch_size=option.batch_size, shuffle=True, num_workers=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, c_graph = load_graph_data("DCRNN_Provide/adj_mx_bay.pkl")
    c_graph = torch.from_numpy(c_graph)
    s_graph = Graph2S(c_graph, option.beta)

    # S1
    # c_graph = c_graph.unsqueeze(0).repeat(torch.cuda.device_count(), 1, 1).to(device=device, dtype=torch.float)
    # s_graph = s_graph.unsqueeze(0).repeat(torch.cuda.device_count(), 1, 1).to(device=device, dtype=torch.float)
    # spatial_graph = c_graph
    # spatial_graph = [c_graph, s_graph]

    # S2
    graph_0 = torch.eye(option.num_of_stations).unsqueeze(0).to(device=device, dtype=torch.float)
    graph_1 = c_graph.unsqueeze(0).to(device=device, dtype=torch.float)
    spatial_graph = torch.cat([graph_0, graph_1 - graph_0], dim=0)

    model = Prediction_Model(Ks=option.num_of_neighbors,
                             encoder_in_channel=option.temporal_in_channel,
                             encoder_out_channel=option.spatial_out_channel,
                             num_stations=option.num_of_stations,
                             switch=option.switch)

    # model = LSTM_Model(option.num_of_stations, 1, 4, 1)
    # model = Merge_ChebNet(2, 96, 6, 1)
    # model = Merge_GCGRU(option.num_of_stations, 1, [8, 8, 1], 2, 3)
    model = STGCN_Model(1, 1, [5, 2], option.num_of_stations, False)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("{} GPU(s) will be used.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model_size(model, 0)

    cudnn.benchmark = True

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, [10], gamma=1)
    option.log = option.save_model

    train(model, training_dataset, validation_dataset, spatial_graph, criterion, optimizer, scheduler, option, device)

    test_main(option.save_model + ".pkl")


def test_main(model_file_name):
    print("Loading Saved Model")
    checkpoint = torch.load(model_file_name)

    model_state_dict = checkpoint["model"]
    option = checkpoint["setting"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_structure = Prediction_Model(Ks=option.num_of_neighbors,
                                       encoder_in_channel=option.temporal_in_channel,
                                       encoder_out_channel=option.spatial_out_channel,
                                       num_stations=option.num_of_stations,
                                       switch=option.switch)

    # model_structure = LSTM_Model(option.num_of_stations, 1, 4, 1)
    # model_structure = Merge_ChebNet(2, 96, 6, 1)
    # model_structure = Merge_GCGRU(option.num_of_stations, 1, [8, 8, 1], 2, 3)
    # model_structure = STGCN_Model(1, 1, [5, 2], option.num_of_stations, False)

    if torch.cuda.device_count() > 1:
        model_structure = nn.DataParallel(model_structure).to(device)
        model_structure.load_state_dict(model_state_dict)
    else:
        model_structure = model_structure.to(device)
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.item()}
        model_structure.load_state_dict(model_state_dict)

    device = torch.device("cpu")

    print("Loading Testing Dataset")
    testing_dataset = DataLoader(Create_Dataset(option.file_name,
                                                mode="test",
                                                dtype=torch.float,
                                                device=device),
                                 batch_size=option.batch_size, num_workers=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, c_graph = load_graph_data("DCRNN_Provide/adj_mx_bay.pkl")
    c_graph = torch.from_numpy(c_graph)
    s_graph = Graph2S(c_graph, option.beta)

    # S1
    c_graph = c_graph.unsqueeze(0).repeat(torch.cuda.device_count(), 1, 1).to(device=device, dtype=torch.float)
    s_graph = s_graph.unsqueeze(0).repeat(torch.cuda.device_count(), 1, 1).to(device=device, dtype=torch.float)
    # spatial_graph = c_graph
    spatial_graph = [c_graph, s_graph]

    # S2
    graph_0 = torch.eye(option.num_of_stations).unsqueeze(0).to(device=device, dtype=torch.float)
    graph_1 = c_graph.unsqueeze(0).to(device=device, dtype=torch.float)
    # spatial_graph = torch.cat([graph_0, graph_1 - graph_0], dim=0)

    criterion = nn.MSELoss(size_average=False)

    mae, mape, rmse, loss = test(model_structure, testing_dataset, spatial_graph, criterion, device)
    print(
        "[ Results ]\n  - loss: {:2.4f}    - mae: {:2.2f}    - mape: {:2.4f}    - rmse: {:2.2f}".format(loss, mae, mape,
                                                                                                        rmse))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    print("GPU ID: ", os.environ["CUDA_VISIBLE_DEVICES"])
    # main()
    test_main("Result_Models/GSTNet.pkl")
