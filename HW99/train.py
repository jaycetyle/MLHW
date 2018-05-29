#!/usr/bin/env python3
import sys
import numpy
import os
import keras
from os import path
from PIL import Image
from PIL import ImageOps
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

IMG_ROWS = 64
IMG_COLS = 64
COLORS = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, COLORS)

BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 32

def load_images(folder):
    x = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not (file.endswith('.jpg') or file.endswith('.jpeg')):
                continue
            img = Image.open(path.join(root, file)).convert('L')
            width, height = img.size
            if width != IMG_COLS or height != IMG_ROWS:
                raise "Incorrect image size: {0}".format(file)
            x.append(numpy.array(img))
            x.append(numpy.array(ImageOps.mirror(img)))
    return x


def load_train(input_path):
    y = []
    x = []

    males = path.join(input_path, "male")
    females = path.join(input_path, "female")
    if not path.exists(females) or not path.exists(males):
        raise "male or female folder not exists"

    females = load_images(females)
    y.extend([0] * len(females))

    males = load_images(males)
    y.extend([1] * len(males))

    x.extend(females)
    x.extend(males)

    x = numpy.array(x)
    y = numpy.array(y)
    x = x.reshape(len(y), IMG_ROWS, IMG_COLS, COLORS)

    return y, x


def main():
    if len(sys.argv) != 3:
        print("Usage: train.py <TRAIN_FOLDER> <MODEL_PATH>")
        return

    print("Parsing taining data")
    y, x = load_train(sys.argv[1])
    y = keras.utils.to_categorical(y, NUM_CLASSES)

    print("Create Model")
    model = Sequential()

    model.add(Conv2D(8, padding='same', kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(32, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(lr=0.005),
                    metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_acc', patience=4)

    model.fit(x, y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[early_stop],
            validation_split=0.25)

    model.save(sys.argv[2])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
