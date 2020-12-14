# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 19:59

@Author : Yuppie
"""
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_fitted_curve(x_range, y_range, model_name, save_file):
    training_file = model_name + ".train.log"
    validation_file = model_name + ".valid.log"
    axis_x = []
    training_data = []
    validation_data = []
    with open(training_file) as ft:
        datas = ft.readlines()
        for data_i in datas:
            axis_x.append(int(data_i.split(",")[0]))
            data = float(data_i.split(",")[1])
            training_data.append(data)

    with open(validation_file) as fv:
        datas = fv.readlines()
        for data_i in datas:
            data = float(data_i.split(",")[1])
            validation_data.append(data)

    plt.figure()
    plt.grid(True, linestyle="--", linewidth=1)

    plt.plot(axis_x, training_data, ls="-", marker=" ", color="r")
    plt.plot(axis_x, validation_data, ls="-", marker=" ", color="g")
    plt.legend(["training loss", "validation loss"], loc="upper right")

    # plt.xticks(np.arange(0, range, int(range/20)))
    plt.xlabel("Training Epoch")
    # plt.yticks(np.arange(0, resolution * 10, resolution))
    plt.ylabel("Loss")
    plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.title("Training and Validation Results")

    plt.savefig(save_file)


if __name__ == "__main__":

    model_name = "Result_Models/ChebNet_1"
    save_file = model_name + ".png"
    plot_fitted_curve([0, 100], [10, 200], model_name, save_file)