#!/usr/bin/env python3
import sys
import numpy
import math
import matplotlib.pyplot as plt

ITEM_COUNT = 18
HISTORY = 9
HOURS_PER_DAY = 24

def parse_train(file):
    file.readline()      # skip header line
    lines = file.readlines()

    row_size = int(HOURS_PER_DAY * len(lines) / ITEM_COUNT)

    # order='F': Use fortan style memory layout (column major)
    data = numpy.zeros((row_size, ITEM_COUNT), order='F')

    for i in range(0, len(lines)):
        line = lines[i].strip().split(',')[3:]
        for hour in range(0, HOURS_PER_DAY):
            val = line[hour]
            col = i % ITEM_COUNT
            row = int(math.floor(i / ITEM_COUNT) * HOURS_PER_DAY + hour)
            data[row, col] = float(val) if val != "NR" else 0.0

    return data


def prepare_yx(data):
    y = data[HISTORY:, HISTORY]

    X = []
    for i in range(0, numpy.size(data, 0) - HISTORY):
        args = data[i:i+HISTORY, :].flatten()   # flatten: 2D to 1D
        X.append(args)

    return y, numpy.array(X)


def predict(X, w, b):
    return b + numpy.dot(X, w)      # y* = (b + Xw)


def gradient(y, X, w, b):
    e = y - predict(X, w, b)                # e = y - y*
    dw = -2 * numpy.dot(X.transpose(), e)   # w = -2 * X'e
    db = -2 * numpy.sum(e)
    return dw, db


def train(y, X, loops=1000, rate = 10**-2):
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


def main():
    if len(sys.argv) != 3:
        print("Usage: q1.py <TRAIN_PATH> <MODEL_PATH>")
        return

    with open(sys.argv[1], encoding="big5") as file:
        data = parse_train(file)

    y, X = prepare_yx(data)
    w, b, loss = train(y, X, 500000)

    plt.plot(loss)
    plt.show()

    with open(sys.argv[2], 'w') as file:
        file.write('{0}\n'.format(b))
        numpy.savetxt(file, w)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)