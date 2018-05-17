#!/usr/bin/env python3
import sys
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

IMG_ROWS = 48
IMG_COLS = 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

BATCH_SIZE = 64
NUM_CLASSES = 7
EPOCHS = 128

def parse_train(file):
    file.readline() # skip header

    y = []
    x = []

    for line in file:
        data = line.split(',')
        y.append(float(data[0]))
        x.extend([float(val) for val in data[1].split()])

    data_count = int(len(x)/IMG_ROWS/IMG_COLS)
    x = numpy.array(x)
    y = numpy.array(y)
    x = x.reshape(data_count, IMG_ROWS, IMG_COLS, 1)

    return y, x


def main():
    if len(sys.argv) != 3:
        print("Usage: train.py <TRAIN_PATH>")
        return

    print("Parsing taining data")
    with open(sys.argv[1], encoding="big5") as file:
        y, x = parse_train(file)
    y = keras.utils.to_categorical(y, NUM_CLASSES)

    print("Create Model")
    model = Sequential()

    model.add(Conv2D(8, padding='same', kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(24, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

    model.fit(x, y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_split=0.1)

    model.save(sys.argv[2])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)