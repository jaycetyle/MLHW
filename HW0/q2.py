#!/usr/bin/env python3
import sys
from PIL import Image

def main():
    if len(sys.argv) != 2:
        print("Usage: q1.py <FILE_PATH>")
        return

    img = Image.open(sys.argv[1]).convert('RGB')
    img = Image.eval(img, lambda i: int(i/2))
    img.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)