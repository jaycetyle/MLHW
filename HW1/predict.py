#!/usr/bin/env python3
import sys
import numpy
import math

ITEM_COUNT = 18
HISTORY = 9

def parse_input(file):
    lines = file.readlines()

    # order='F': Use fortan style memory layout (column major)
    data = numpy.zeros((math.floor(len(lines) / ITEM_COUNT), HISTORY*ITEM_COUNT), order='F')

    for i in range(0, len(lines)):
        line = lines[i].strip().split(',')[2:]
        for num in range(0, HISTORY):
            val = line[num]
            row = math.floor(i / ITEM_COUNT)
            col = int(num * ITEM_COUNT + i % ITEM_COUNT)
            data[row, col] = float(val) if val != "NR" else 0.0

    return data


def load_model(file):
    b = float(file.readline().strip())
    w = numpy.loadtxt(file)
    return w, b


def predict(X, w, b):
    return b + numpy.dot(X, w)      # y* = (b + Xw)


def write_output(file, y):
    file.write("id,value\n")
    for i in range(0, numpy.size(y, 0)):
        file.write("id_{0},{1}\n".format(i, y[i]))


def main():
    if len(sys.argv) != 4:
        print("Usage: predict.py <INPUT_PATH> <MODEL_PATH> <OUTPUT_PATH>")
        return

    with open(sys.argv[1], encoding="big5") as file:
        X = parse_input(file)

    print(X)

    with open(sys.argv[2], encoding="big5") as file:
        w, b = load_model(file)

    y = predict(X, w, b)

    with open(sys.argv[3], 'w') as file:
        write_output(file, y)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)