# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 19:08
@Author : Yuppie
"""
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from Normalize import Switch_Norm_1D


class Cheb_Transform(object):
    def __init__(self):
        pass

    @staticmethod
    def transform(inputs, L, K):
        batch_size, num_stations, in_channels = inputs.size()
        outputs = torch.zeros([batch_size, num_stations, in_channels, K], device=inputs.device, dtype=torch.float)
        if K == 1:
            outputs[:, :, :, 0] = inputs
        elif K == 2:
            L = L.repeat(batch_size, 1, 1)  # batch_size * N * N
            outputs[:, :, :, 0] = inputs
            outputs[:, :, :, 1] = torch.bmm(L, inputs)
        else:
            L = L.repeat(batch_size, 1, 1)
            outputs[:, :, :, 0] = inputs
            outputs[:, :, :, 1] = torch.bmm(L, inputs)
            for k in range(2, K):
                outputs[:, :, :, k] = 2 * torch.bmm(L, outputs[:, :, :, k - 1]) - outputs[:, :, :, k - 2]

        return outputs


class Graph2L(object):
    def __init__(self):
        pass

    @staticmethod
    def g2L(Graph):
        D = torch.diag(torch.sum(Graph, dim=1) ** (-1 / 2))
        L = torch.eye(Graph.size(0), device=Graph.device, dtype=Graph.dtype) - torch.mm(torch.mm(D, Graph), D)
        return L


class graph_conv(nn.Module):
    def __init__(self, K, in_channel, out_channel, bias=True):
        super(graph_conv, self).__init__()

        self.weight = nn.ModuleList([nn.Linear(K, out_channel, False) for _ in range(in_channel)])
        # init.xavier_normal_(self.weight[d].weight for d in range(in_channel))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channel))
            init.normal_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.K = K

    def forward(self, inputs, L):
        L = L.to(device=inputs.device, dtype=inputs.dtype)

        inputs = Cheb_Transform.transform(inputs, L, self.K)  # B * N * D * K

        outputs = sum([self.weight[d](inputs[:, :, d, :]) for d in range(self.in_channel)])

        if self.bias is not None:
            return outputs + self.bias
        else:
            return outputs


class ChebNet(nn.Module):
    def __init__(self, K, in_channel, hid_channel, out_channel):
        super(ChebNet, self).__init__()

        self.gc1 = graph_conv(K, in_channel, hid_channel)
        self.gc2 = graph_conv(K, hid_channel, hid_channel)
        # self.gc3 = graph_conv(K, hid_channel, out_channel)

        self.norm_1 = Switch_Norm_1D(hid_channel)
        # self.norm_2 = Switch_Norm_1D(hid_channel)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs, Graph):
        Graph = Graph if Graph.dim() == 2 else Graph.squeeze(0)

        L = Graph2L.g2L(Graph)
        outputs = self.act(self.norm_1(self.gc1(inputs, L)))
        # outputs = self.act(self.norm_2(self.gc2(outputs, L)))
        # outputs = self.act(self.gc3(outputs, L))

        outputs = self.act(self.gc2(outputs, L))

        return outputs


class Merge_ChebNet(nn.Module):
    def __init__(self, K, in_channel, hid_channel, out_channel):
        super(Merge_ChebNet, self).__init__()

        self.gcn = ChebNet(K, in_channel, hid_channel, out_channel)

    def forward(self, inputs, graph):
        batch_size, num_stations = inputs.size(0), inputs.size(1)

        graph = graph if graph.dim() == 2 else graph.squeeze(0)

        # recent_data = inputs[:, :, -6:, :]  # batch * N * 6 * 1
        # period_data = inputs[:, :, ::96, :]  # batch * N * 2 * 1

        # inputs = torch.cat([recent_data, period_data], dim=-2)  # B * N * 8 * 2
        # inputs = inputs.view(batch_size, num_stations, -1)  # B * N * 16

        outputs = self.gcn(inputs.view(batch_size, num_stations, -1), graph).unsqueeze(-2)  # B * N * 1 * out_channel

        return outputs
