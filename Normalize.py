# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 19:07

@Author : Yuppie
"""
import torch
import torch.nn as nn


class Switch_Norm_1D(nn.Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(Switch_Norm_1D, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma

        self.weight = nn.Parameter(torch.ones(1, 1, in_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, in_channels))

        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, in_channels, 1))
            self.register_buffer('running_var', torch.zeros(1, in_channels, 1))

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    @staticmethod
    def _check_input_dim(inputs):
        if inputs.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(inputs.dim()))

    def forward(self, inputs):
        Switch_Norm_1D._check_input_dim(inputs)

        inputs = inputs.transpose(1, 2)

        mean_in = inputs.mean(-1, keepdim=True)
        var_in = inputs.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        inputs = (inputs - mean) / (var + self.eps).sqrt()
        inputs = inputs.transpose(1, 2)

        return self.weight * inputs + self.bias


class Switch_Norm_2D(nn.Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(Switch_Norm_2D, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, 1, 1, in_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, in_channels))

        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, in_channels, 1))
            self.register_buffer('running_var', torch.zeros(1, in_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    @staticmethod
    def _check_input_dim(inputs):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(inputs.dim()))

    def forward(self, inputs):
        self._check_input_dim(inputs)
        batch_size, num_stations, seq_len, in_channels = inputs.size()
        inputs = inputs.transpose(-2, -1).transpose(-3, -2)
        inputs = inputs.contiguous().view(batch_size, in_channels, -1)

        mean_in = inputs.mean(-1, keepdim=True)
        var_in = inputs.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        inputs = (inputs - mean) / (var + self.eps).sqrt()
        inputs = inputs.contiguous().view(batch_size, in_channels, num_stations, seq_len)
        inputs = inputs.transpose(-3, -2).transpose(-2, -1)

        return self.weight * inputs + self.bias