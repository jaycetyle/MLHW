#!/usr/bin/env python3
import sys
import numpy
from PIL import Image
from keras.models import load_model

IMG_ROWS = 48
IMG_COLS = 48

sentiment = ["生氣", "厭惡", "恐懼", "高興", "難過", "驚訝", "無表情"]

def load_img(path):
    img = Image.open(path).convert('L')
    img = img.resize((IMG_ROWS, IMG_COLS), Image.ANTIALIAS)
    x = numpy.array(img)
    return x.reshape(1, IMG_ROWS, IMG_COLS, 1)


def main():
    if len(sys.argv) != 3:
        print("Usage: predict.py <MODEL_PATH> <IMAGE>")
        return

    input = load_img(sys.argv[2])

    model = load_model(sys.argv[1])
    label = model.predict_classes(input, verbose=0)

    print("ANS:", sys.argv[2], sentiment[label[0]])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

