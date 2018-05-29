#!/usr/bin/env python3
import sys
import os
from os import path
from PIL import Image

SUB_FOLDERS = 128
IMAGE_SIZE = (64, 64)

def preproc(inpath, outpath):
    img = Image.open(inpath).convert('L')
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)

    filename = path.basename(inpath)
    outdir = path.join(outpath, "{:0>3}".format(hash(filename) % SUB_FOLDERS))
    if not path.exists(outdir):
        os.makedirs(outdir)

    img.save(path.join(outdir, filename))


def main():
    inputs = sys.argv[1]
    outputs = sys.argv[2]

    for root, dirs, files in os.walk(inputs):
        for file in files:
            if not (file.endswith('.jpg') or file.endswith('.jpeg')):
                continue
            filepath = path.join(root, file)
            print(filepath)
            try:
                preproc(filepath, outputs)
            except Exception as e:
                print("failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)