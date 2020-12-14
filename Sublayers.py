# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/17 10:08

@Author : Yuppie
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


from Compare_Methods.ChebNet import ChebNet


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        results = super(CausalConv1d, self).forward(inputs)
        padding = self.padding[0]
        if padding != 0:
            return results[:, :, :-padding]
        return results


class General_Attention(nn.Module):
    def __init__(self, input_dim):
        super(General_Attention, self).__init__()
        self.scale = np.power(input_dim, 0.5)
        self.softmax = F.softmax
        self.dropout = nn.Dropout(0.1)

    def forward(self, Query, Key, Value, attn_mask=None):
        # attention shape: (batch_size, new_len, len)
        attention = torch.bmm(Query, Key.transpose(1, 2)) / self.scale

        if attn_mask is not None:
            assert attention.size() == attn_mask.size()

            attention = attention * attn_mask
            attention.data.masked_fill_(torch.eq(attention, 0), -float("inf"))

        attention = self.softmax(attention, dim=2)
        outputs = torch.bmm(attention, Value)

        return outputs, attention


class Inception_Temporal_Layer(nn.Module):
    def __init__(self, num_stations, In_channels, Hid_channels, Out_channels):
        super(Inception_Temporal_Layer, self).__init__()
        self.temporal_conv1 = CausalConv1d(In_channels, Hid_channels, 3, dilation=1, groups=1)
        # init.xavier_normal_(self.temporal_conv1.weight)

        self.temporal_conv2 = CausalConv1d(Hid_channels, Hid_channels, 2, dilation=2, groups=1)
        # init.xavier_normal_(self.temporal_conv2.weight)

        self.temporal_conv3 = CausalConv1d(Hid_channels, Hid_channels, 2, dilation=4, groups=1)
        # init.xavier_normal_(self.temporal_conv3.weight)

        self.conv1_1 = CausalConv1d(3 * Hid_channels, Out_channels, 1)

        self.num_stations = num_stations
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        # (Batch_size, Number_Station, Seq_len, In_channel)
        output_1 = torch.cat([self.temporal_conv1(inputs[:, s_i].transpose(1, 2)).transpose(1, 2).unsqueeze(1)
                              for s_i in range(self.num_stations)], dim=1)
        output_1 = self.act(output_1)

        output_2 = torch.cat([self.temporal_conv2(output_1[:, s_i].transpose(1, 2)).transpose(1, 2).unsqueeze(1)
                              for s_i in range(self.num_stations)], dim=1)
        output_2 = self.act(output_2)

        output_3 = torch.cat([self.temporal_conv3(output_2[:, s_i].transpose(1, 2)).transpose(1, 2).unsqueeze(1)
                              for s_i in range(self.num_stations)], dim=1)
        output_3 = self.act(output_3)

        outputs = torch.cat([output_1, output_2, output_3], dim=-1)

        outputs = torch.cat([self.conv1_1(outputs[:, s_i].transpose(1, 2)).transpose(1, 2).unsqueeze(1)
                             for s_i in range(self.num_stations)], dim=1)

        return outputs


class Spatial_Feature_Transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spatial_Feature_Transform, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

        init.xavier_normal_(self.fc1.weight)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        return outputs


class Correlation_Metric(nn.Module):
    def __init__(self, num_stations, in_channels, hidden_channels, switch='gaussian'):
        super(Correlation_Metric, self).__init__()
        self.f = F.softmax
        self.num_stations = num_stations
        self.in_channels = in_channels
        self.switch = switch

        self.linear = Spatial_Feature_Transform(in_channels, hidden_channels)

    def forward(self, inputs, graph):
        inputs = self.linear(inputs)

        outputs = torch.bmm(inputs, inputs.transpose(1, 2)) * graph
        if self.switch == 'gaussian':
            outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))
            return self.f(outputs, dim=2)
        elif self.switch == 'dot_product':
            return outputs / self.num_stations


class Non_local_gcn(nn.Module):
    def __init__(self, K, in_channels, out_channels, num_stations, switch='gaussian'):
        super(Non_local_gcn, self).__init__()
        self.gcn = ChebNet(K, in_channels, out_channels, out_channels)
        self.correlation = Correlation_Metric(num_stations, out_channels, out_channels, switch)

        self.act = nn.LeakyReLU()

    def forward(self, inputs, c_graph, s_graph):
        local_output = self.act(self.gcn(inputs, c_graph))
        correlation = self.correlation(local_output, s_graph)

        return torch.bmm(correlation, local_output) + inputs


class Local_gcn(nn.Module):
    def __init__(self, K, in_channels, out_channels):
        super(Local_gcn, self).__init__()
        self.gcn = ChebNet(K, in_channels, out_channels, out_channels)

        self.act = nn.LeakyReLU()

    def forward(self, inputs, c_graph, s_graph):
        local_output = self.gcn(inputs, c_graph)

        return self.act(local_output)


def Graph2S(input_graph, beta):
    S_Graph = [[beta if input_graph[i, j] > 0 else 1 for i in range(input_graph.size(0))] for j in range(input_graph.size(1))]

    for i in range(len(S_Graph)):
            S_Graph[i][i] = 0.
    S_Graph = torch.from_numpy(np.array(S_Graph))

    return S_Graph