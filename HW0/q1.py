#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: q1.py <FILE_PATH>")
        return

    with open(sys.argv[1]) as file:
        words = file.readline().split()

    counts = {}
    word_list = []
    for word in words:
        if word not in counts:
            counts[word] = 1
            word_list.append(word)
        else:
            counts[word] += 1

    for i in range(0, len(word_list)):
        word = word_list[i]
        print("{0} {1} {2}".format(word, i, counts[word]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)