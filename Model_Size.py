# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 17:27

@Author : Yuppie
"""
import numpy as np


def model_size(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:.4f} MB'.format(model._get_name(), para * type_size / 1000 / 1000))
