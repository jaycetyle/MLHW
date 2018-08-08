#!/usr/bin/env python3
import sys
import numpy
import keras
import gensim
from keras.models import Sequential
from keras.layers import Embedding,  Dense, Bidirectional, LSTM
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

SPLIT_TOKEN = ' +++$+++ '
WORD_VECTOR_LENGTH = 192
BATCH_SIZE = 64
EPOCHS = 12
NUM_CLASSES = 2

def parse_label_data(file):
    y = []
    x = []

    for line in file:
        data = line.split(SPLIT_TOKEN)
        y.append(int(data[0]))
        x.append(data[1].strip())

    return numpy.array(y), numpy.array(x)


def train_gensim(x):
    gensim_input = [line.split(" ") for line in x]
    model = gensim.models.Word2Vec(gensim_input, size=WORD_VECTOR_LENGTH, min_count=1, workers=4)
    return model


def main():
    if len(sys.argv) != 4:
        print("Usage: train.py <TRAIN_PATH> <DICT_PATH> <MODEL_PATH>")
        return

    print("Parsing taining data")
    with open(sys.argv[1]) as file:
        y, lines = parse_label_data(file)
    y = keras.utils.to_categorical(y, NUM_CLASSES)

    print("Create gensim word to vector model")
    w2v_model = train_gensim(lines)

    print("Create dictionary")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    dict_len = len(tokenizer.word_index) + 1

    numpy.save(sys.argv[2], tokenizer.word_index)

    print("Create sequence")
    max_length = len(max(lines, key=len))                   # find the length of the longest sequence in lines
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = pad_sequences(sequences, maxlen=max_length)

    print("Create Embedding Layer Model")
    embedding_weight = numpy.zeros((dict_len, WORD_VECTOR_LENGTH))
    for word, index in tokenizer.word_index.items():
        try:
            embedding_weight[index] = w2v_model.wv[word]
        except:
            print(" '{0}' is not in vocabulary".format(word))

    print("Create Model")
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(dict_len,
                        WORD_VECTOR_LENGTH,
                        weights=[embedding_weight],
                        input_length=max_length,
                        input_shape=(max_length,),
                        trainable=False))

    # RNN Layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.2))

    # Full Connected Layer
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile and train
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(sequences,
              y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_split=0.1)
    model.save(sys.argv[3])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
