#!/usr/bin/env python3
import sys
import numpy
import matplotlib.pyplot as plt

workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education = ["Bachelors", "Some-college", "11th", "HS-grad",
    "Prof-school", "Assoc-acdm", "Assoc-voc",
    "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
    "Doctorate", "5th-6th", "Preschool"]
marital_status = ["Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support", "Craft-repair", "Other-service",
    "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"]
race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"]
sex = ["Female", "Male"]
native_country = ["United-States", "Cambodia", "England",
    "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
    "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
    "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
    "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
    "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
    "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]


def parse_data(data, table = None):
    if table is None:
        return [int(data)]
    if (data in table):
        index = table.index(data)
    else:
        index = len(table)
    ret = [0] * (len(table) + 1)
    ret[index] = 1
    return ret


def parse_train(file):
    X = []
    y = []
    for line in file:
        man = line.split(',')
        if len(man) < 15:
            continue
        man = [col.strip() for col in man]
        x = []
        x.extend(parse_data(man[0]))
        x.extend(parse_data(man[1], workclass))
        x.extend(parse_data(man[2]))
        x.extend(parse_data(man[3], education))
        x.extend(parse_data(man[4]))
        x.extend(parse_data(man[5], marital_status))
        x.extend(parse_data(man[6], occupation))
        x.extend(parse_data(man[7], relationship))
        x.extend(parse_data(man[8], race))
        x.extend(parse_data(man[9], sex))
        x.extend(parse_data(man[10]))
        x.extend(parse_data(man[11]))
        x.extend(parse_data(man[12]))
        x.extend(parse_data(man[13], native_country))
        if (man[14] == "<=50K"):
            y.append(0)
        else:
            y.append(1)
        X.append(x)
    return numpy.array(y), numpy.array(X)


def predict(X, w, b):
    yp = 1.0 / (1.0 + numpy.exp(-(b + numpy.dot(X, w))))   # y* = sigmoid(z)
    return numpy.clip(yp,0.00000000000001,0.99999999999999)


def gradient(y, X, w, b):
    yp = predict(X, w, b)
    e = y - yp                          # e = y - y*
    dw = -numpy.dot(X.transpose(), e)   # w = -X'e
    db = -numpy.sum(e)
    return dw, db


def train(y, X, loops=1000, rate = 10**-4):
    cnt = numpy.shape(y)[0]
    w = numpy.zeros(numpy.shape(X)[1])
    b = 0

    sum_dw2 = sum_db2 = 0
    err = []

    for i in range(0, loops):
        dw, db = gradient(y, X, w, b)
        w = w - rate * dw
        b = b - rate * db
        if i % 100 == 0:
            yp = predict(X, w, b)
            loss = -(numpy.dot(y, numpy.log(yp)) + numpy.dot((1-y), numpy.log(1-yp)))
            err.append(loss)
            errate = numpy.sum(numpy.absolute(y - numpy.around(yp))) / float(cnt)
            print("{0} err: {1:0.2%}, loss: {2}".format(i, errate, loss))
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

    max = numpy.clip(numpy.max(numpy.abs(X), axis=0),1,100000000)
    X = X / max
    w, b, loss = train(y, X, 50000)

    plt.plot(loss)
    plt.show()

    with open(sys.argv[2], 'w') as file:
        save_model(file, w, b)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)