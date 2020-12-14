# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 13:53

@Author : Yuppie
"""
import numpy as np


class Evaluation_Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def MAE(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def MAPE(target, output):
        return np.mean(np.abs(target - output) / (target + 5))

    @staticmethod
    def RMSE(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation_Utils.MAE(target, output)
        mape = Evaluation_Utils.MAPE(target, output)
        rmse = Evaluation_Utils.RMSE(target, output)

        return mae, mape, rmse
