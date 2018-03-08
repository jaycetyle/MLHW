#!/usr/bin/env python3
import sys
import numpy
import math
import matplotlib.pyplot as plt

HISTORY = 9
ITEM_COUNT = 18
HOURS_PER_DAY = 24

DATA_DAYS = 20
DATA_MONTHS = 12


def month_data_to_yx(data):
    y = data[HISTORY:, HISTORY]

    X = []
    for i in range(0, numpy.size(data, 0) - HISTORY):
        args = data[i:i+HISTORY, :].flatten()   # flatten: 2D to 1D
        X.append(args)

    return numpy.array(y), numpy.array(X)


def parse_train(file):
    file.readline()      # skip header line

    y = numpy.array([])
    X = numpy.array([]).reshape(0, HISTORY * ITEM_COUNT)    # 0x162

    for m in range(0, DATA_MONTHS):
        month_data = numpy.array([]).reshape(0, ITEM_COUNT)
        for d in range(0, DATA_DAYS):
            day_data = []
            for l in range(0, ITEM_COUNT):
                line = file.readline().strip().split(',')[3:]
                line = [float(val) if val != "NR" else 0.0 for val in line]
                day_data.append(line)

            day_data = numpy.array(day_data).transpose()
            month_data = numpy.concatenate((month_data, day_data), axis = 0)

        month_y, month_X = month_data_to_yx(month_data)
        y = numpy.concatenate((y, month_y), axis = 0)
        X = numpy.concatenate((X, month_X), axis = 0)

    return numpy.array(y), X


def fold_validate(y, X, n=5):
    train_y = numpy.array([])
    train_X = numpy.array([]).reshape(0, HISTORY * ITEM_COUNT)
    valid_y = numpy.array([])
    valid_X = numpy.array([]).reshape(0, HISTORY * ITEM_COUNT)

    for i in range(0, numpy.size(y), n):
        train_y = numpy.concatenate((train_y, y[i:i+n-1]), axis = 0)
        train_X = numpy.concatenate((train_X, X[i:i+n-1, :]), axis = 0)
        valid_y = numpy.concatenate((valid_y, y[i+n:i+n+1]), axis = 0)
        valid_X = numpy.concatenate((valid_X, X[i+n:i+n+1, :]), axis = 0)

    return train_y, train_X, valid_y, valid_X


def predict(X, w, b):
    return b + numpy.dot(X, w)      # y* = (b + Xw)


def gradient(y, X, w, b):
    e = y - predict(X, w, b)                # e = y - y*
    dw = -2 * numpy.dot(X.transpose(), e)   # w = -2 * X'e
    db = -2 * numpy.sum(e)
    return dw, db


def train(y, X, loops=1000, rate = 0.01):
    w = numpy.zeros(HISTORY * ITEM_COUNT)
    b = 0

    sum_dw2 = sum_db2 = 0
    err = []

    for i in range(0, loops):
        dw, db = gradient(y, X, w, b)
        sum_dw2 += dw*dw
        sum_db2 += db*db
        w = w - rate * dw / numpy.sqrt(sum_dw2)   # adagrad
        b = b - rate * db / numpy.sqrt(sum_db2)
        if i % 100 == 0:
            err.append(numpy.average(numpy.absolute(y - predict(X, w, b))))
    return w, b, err


def save_model(file, w, b):
    file.write('{0}\n'.format(b))
    numpy.savetxt(file, w)


def main():
    if len(sys.argv) != 3:
        print("Usage: train.py <TRAIN_PATH> <MODEL_PATH>")
        return

    with open(sys.argv[1], encoding="big5") as file:
        y, X = parse_train(file)

    print(X)
    # train_y, train_X, valid_y, valid_X = fold_validate(y, X)
    # w, b, loss = train(train_y, train_X, 10000)

    # ye = numpy.absolute(train_y - predict(train_X, w, b))
    # print("train: avg = {0}, std = {1}".format(numpy.average(ye), numpy.std(ye)))
    # ye = numpy.absolute(valid_y - predict(valid_X, w, b))
    # print("valid: avg = {0}, std = {1}".format(numpy.average(ye), numpy.std(ye)))

    w, b, loss = train(y, X, 1000)
    ye = numpy.absolute(y - predict(X, w, b))

    print("train: avg = {0}, std = {1}".format(numpy.average(ye), numpy.std(ye)))

    plt.plot(loss)
    plt.show()

    with open(sys.argv[2], 'w') as file:
        save_model(file, w, b)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)