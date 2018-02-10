#!/usr/bin/env python3
import sys
from PIL import Image

def main():
    if len(sys.argv) != 2:
        print("Usage: q1.py <FILE_PATH>")
        return

    img = Image.open(sys.argv[1])
    pixels = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r, g, b = img.getpixel((i, j))
            pixels[i,j] = (int(r/2), int(g/2), int(b/2))
    
    img.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)