# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 13:50

@Author : Yuppie
"""
import numpy as np
import torch
import time
from Utils import Evaluation_Utils


def get_performance(preds, targets, datasets):
    try:
        Dataset = datasets.dataset
    except:
        Dataset = datasets

    scaler = Dataset.scaler

    preds = scaler.inverse_transform(preds.cpu().numpy())
    targets = scaler.inverse_transform(targets.cpu().numpy())

    mae, mape, rmse = Evaluation_Utils.total(targets.reshape(-1), preds.reshape(-1))

    return mae, mape, rmse


def test(model, dataset, graph, criterion, device):
    # TODO: recover data and then compute the prediction error

    total_loss = 0.0
    MAE = []
    MAPE = []
    RMSE = []

    with torch.no_grad():
        for data in dataset:
            # preds shape  : (batch_size, num_stations, 1, input_dim)
            # targets shape: (batch_size, num_stations, 1, input_dim)
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs, graph)

            # shape : (num_stations, batch_size, input_dim)
            targets = targets.transpose(0, 2).squeeze(0)
            preds = preds.transpose(0, 2).squeeze(0)

            loss = criterion(preds, targets)
            total_loss += loss.item()

            mae, mape, rmse = get_performance(preds, targets, dataset)

            MAE += [mae]
            MAPE += [mape]
            RMSE += [rmse]

    return np.mean(MAE), np.mean(MAPE), np.mean(RMSE), total_loss / (2 * len(dataset.dataset))


def train_epoch(model, training_data, graph, criterion, optimizer, device):
    total_loss = 0.0
    i = 0
    for data in training_data:
        # prepare data
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)

        model.zero_grad()

        # forward
        preds = model(inputs, graph)

        # backward
        loss = criterion(preds, target)
        loss.backward()

        # print("==" * 40)
        # print(i, torch.sum(target), torch.sum(preds), torch.sum((target - preds) * (target - preds)))

        # update parameters
        optimizer.step()

        total_loss += loss.item()
        i += 1
        # print("  - one batch data time: {:2.2f} min".format((time.time() - start)/60))

    return model, total_loss / (2 * len(training_data.dataset))


def eval_epoch(model, validation_data, graph, criterion, device):
    total_loss = 0.0

    with torch.no_grad():
        for data in validation_data:
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)

            preds = model(inputs, graph)

            loss = criterion(preds, target)

            total_loss += loss.item()

    return total_loss / (2 * len(validation_data.dataset))


def train(model, training_data, validation_data, graph, criterion, optimizer, scheduler, option, device):
    log_train_file = None
    log_valid_file = None

    if option.log:
        log_train_file = option.log + ".train.log"
        log_valid_file = option.log + ".valid.log"

        print("[ INFO ] Training performance will be written to file\n {:s} and {:s}".format(
            log_train_file, log_valid_file))

    valid_losses = []

    for each_epoch in range(option.epoch):
        print("[ Epoch {:d} ]".format(each_epoch))
        scheduler.step()
        start_time = time.time()
        model, train_loss = train_epoch(model, training_data, graph, criterion, optimizer, device)
        print("  - (Training) loss: {:2.4f},  elapse: {:2.2f} min".format(
            train_loss,
            (time.time() - start_time) / 60))

        start_time = time.time()
        eval_loss = eval_epoch(model, validation_data, graph, criterion, device)
        print("  - (Validation) loss: {:2.4f},  elapse: {:2.2f} min".format(
            eval_loss,
            (time.time() - start_time) / 60))

        valid_losses += [eval_loss]

        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "setting": option,
            "epoch": each_epoch
        }

        if option.save_model:
            if option.save_mode == "best":
                model_name = option.save_model + ".pkl"
                if eval_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print("  - [ INFO ] The checkpoint file has been updated.")
            elif option.save_mode == "all":
                model_name = option.save_model + "_loss_{:2.4f}.pkl".format(eval_loss)
                torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as train_file, open(log_valid_file, "a") as valid_file:
                train_file.write("{}, {:2.4f}\n".format(each_epoch, train_loss))
                valid_file.write("{}, {:2.4f}\n".format(each_epoch, eval_loss))
