#!/usr/bin/env python3
import sys
import numpy
import os
from os import path
from PIL import Image
from keras.models import load_model

IMG_ROWS = 64
IMG_COLS = 64

gender = ["女", "男"]

def predict(model, file):
    if not (file.endswith('.jpg') or file.endswith('.jpeg')):
        return
    img = Image.open(file).convert('L')
    img = img.resize((IMG_ROWS, IMG_COLS), Image.BILINEAR)
    x = numpy.array(img).reshape(1, IMG_ROWS, IMG_COLS, 1)
    label = model.predict_classes(x, verbose=0)

    print("{0} 是 {1} 的".format(file, gender[label[0]]))

def main():
    if len(sys.argv) != 3:
        print("Usage: predict.py <MODEL_PATH> <FOLDER>")
        return

    print("Load model")
    model = load_model(sys.argv[1])

    for root, dirs, files in os.walk(sys.argv[2]):
        for file in files:
            predict(model, path.join(root, file))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

