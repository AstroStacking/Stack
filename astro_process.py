#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import scipy as sp
from PIL import Image
import skimage
import sklearn

if __name__ == "__main__":
    import argparse
    import os
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process Astrophotography')
    parser.add_argument('--input', type=str, required=True, help='Input image')
    parser.add_argument('--output', type=str, required=True, help='Output image name')
    parser.add_argument('--root-power', type=float, help='Histogram stretch factor')
    parser.set_defaults(root_power=6.)

    args = parser.parse_args()

    logging.info(f"Reading images {args.input}")
    img = Image.open(args.input)
    
    average = np.power(img, args.root_power)

    logging.info(f"Saving output file {args.output}")
    skimage.io.imsave(args.output, (average * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
