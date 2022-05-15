#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import skimage

def readFiles(images):
    imgs = [Image.open(filename) for filename in filenames]
    imgs = [np.asarray(img) / np.iinfo(np.asarray(img).dtype).max for img in imgs]
    return imgs

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Stack Astrophotography')
    parser.add_argument('images', type=str, nargs='+', help='images to stack')
    parser.add_argument('--output', type=str, help='Output image name')

    args = parser.parse_args()
    filenames = args.images

    r = np.max(readFiles(filenames), axis=0)
    skimage.io.imsave(args.output, (r * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
