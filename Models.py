# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/17 10:08

@Author : Yuppie
"""
import torch
import torch.nn as nn
from Sublayers import General_Attention, Inception_Temporal_Layer, Non_local_gcn, Local_gcn
from Normalize import Switch_Norm_2D


class Prediction_Model(nn.Module):
    def __init__(self, Ks, encoder_in_channel, encoder_out_channel, num_stations, switch):
        super(Prediction_Model, self).__init__()
        self.encoder = Prediction_Encoder(Ks, encoder_in_channel, encoder_out_channel, num_stations, switch)
        self.decoder = Prediction_Decoder(encoder_out_channel, encoder_in_channel)

    def forward(self, inputs, graph):
        st_outputs = self.encoder(inputs, graph[0], graph[1])
        predictions = self.decoder(inputs=st_outputs[:, :, -1, :].unsqueeze(-2),
                                   key=st_outputs[:, :, :-1, :],
                                   value=inputs[:, :, 1:, :])

        return predictions


class Prediction_Encoder(nn.Module):
    def __init__(self, K, in_channels, out_channels, num_stations,  switch='gaussian'):
        super(Prediction_Encoder, self).__init__()
        self.tc_1 = Inception_Temporal_Layer(num_stations, in_channels, 4*in_channels, out_channels)
        self.sa_1 = Non_local_gcn(K, out_channels, out_channels, num_stations, switch)
        # self.sa_1 = Local_gcn(K, out_channels, out_channels)

        self.tc_2 = Inception_Temporal_Layer(num_stations, out_channels, out_channels, out_channels)
        self.sa_2 = Non_local_gcn(K, out_channels, out_channels, num_stations, switch)
        # self.sa_2 = Local_gcn(K, out_channels, out_channels)

        # self.conv1_1 = CausalConv1d(in_channels, out_channels, 1)

        self.norm_1 = Switch_Norm_2D(out_channels)
        self.norm_2 = Switch_Norm_2D(out_channels)
        self.norm_3 = Switch_Norm_2D(out_channels)
        self.norm_4 = Switch_Norm_2D(out_channels)

        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        self.num_stations = num_stations
        self.in_channels = in_channels

    def forward(self, inputs, c_graph, s_graph):
        c_graph = c_graph if c_graph.dim() == 3 else c_graph.squeeze(0)
        s_graph = s_graph if s_graph.dim() == 3 else s_graph.squeeze(0)

        batch_size, num_stations, seq_len, temporal_in_channel = inputs.size()

        assert num_stations == self.num_stations
        assert temporal_in_channel == self.in_channels

        # inputs = torch.cat([self.conv1_1(inputs[:, s_i].transpose(1, 2)).transpose(1, 2).unsqueeze(1)
        #                     for s_i in range(self.num_stations)], dim=1)

        temporal_feature = self.norm_1(self.tc_1(inputs))
        spatial_feature = torch.cat([self.sa_1(temporal_feature[:, :, i], c_graph, s_graph).unsqueeze(2) for i in range(seq_len)], dim=2)
        spatial_feature = self.act(self.norm_2(spatial_feature))

        temporal_feature = self.norm_3(self.tc_2(spatial_feature))
        spatial_feature = torch.cat([self.sa_2(temporal_feature[:, :, i], c_graph, s_graph).unsqueeze(2) for i in range(seq_len)], dim=2)
        spatial_feature = self.act(self.norm_4(spatial_feature))

        return spatial_feature


class Prediction_Decoder(nn.Module):
    def __init__(self, spatial_out_channel, temporal_in_channel):
        super(Prediction_Decoder, self).__init__()
        self.attention = General_Attention(spatial_out_channel)
        self.hidden_dim = spatial_out_channel

        self.W_q = nn.Linear(spatial_out_channel, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(spatial_out_channel, self.hidden_dim, bias=False)
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.xavier_normal_(self.W_k.weight)

    def forward(self, inputs, key, value):
        batch_size, num_stations, new_len, spatial_out_channel = inputs.size()
        batch_size, num_stations, seq_len, temporal_in_channel = value.size()

        inputs = self.W_q(inputs).view(-1, new_len, self.hidden_dim)
        key = self.W_k(key).view(-1, seq_len, self.hidden_dim)

        value = value.view(-1, seq_len, temporal_in_channel)

        outputs, _ = self.attention(inputs, key, value)
        outputs = torch.cat([batch_i.unsqueeze(0) for batch_i in torch.chunk(outputs, batch_size, dim=0)], dim=0)
        return outputs
