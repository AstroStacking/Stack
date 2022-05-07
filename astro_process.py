#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import scipy as sp
from PIL import Image
import skimage
import sklearn

NB_BINS = 65536

def histogramSmoothing(channel, hist_smoothing, sky_level_factor):
    hist0, bins0 = np.histogram(channel, bins=NB_BINS, range=(0, 1))
    filter = np.ones((hist_smoothing), dtype=np.float64) / hist_smoothing
    histfiltered = np.convolve(hist0, filter, mode="full")
    histmax = np.argmax(histfiltered[hist_smoothing:-hist_smoothing]) + hist_smoothing
    skyvalue = histfiltered[histmax] * sky_level_factor
    tmparray = histfiltered[:histmax]
    skybin = tmparray.shape[0] - np.argmax((tmparray < skyvalue)[::-1]) - 1
    channel = (channel - skybin/NB_BINS) / (1-skybin/NB_BINS)
    return channel

def deconvolve(img, radius, sigma):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    weights /= 2 * np.pi * sigma * sigma

    return skimage.restoration.richardson_lucy(img, weights[:,:,None])

if __name__ == "__main__":
    import argparse
    import os
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process Astrophotography')
    parser.add_argument('--input', type=str, required=True, help='Input image name')
    parser.add_argument('--output', type=str, required=True, help='Output image name')

    parser.add_argument('--deconvolution-output', type=str, help='Output image name after deconvolution')
    parser.add_argument('--power-adjustment-output', type=str, help='Output image name after power adjustment')
    parser.add_argument('--sky-removal-output', type=str, help='Output image name after sky removal')

    parser.add_argument('--radius', type=int, help='Gaussian deconvolve kernel radius')
    parser.add_argument('--sigma', type=int, help='Gaussian deconvolve kernel sigma')
    parser.add_argument('--root-power', type=float, help='Histogram stretch factor')
    parser.add_argument('--hist-smoothing', type=int, help='Histogram smoothing')
    parser.add_argument('--sky-level-factor', type=float, help='Factor for the sky level')
    parser.add_argument('--sky-removal-passes', type=int, help='Number of sky removal passes')
    parser.set_defaults(root_power=0.6, sky_level_factor=0.6, hist_smoothing=601, sky_removal_passes=2, radius=5, sigma=1)

    args = parser.parse_args()

    logging.info(f"Reading images {args.input}")
    img = Image.open(args.input)
    img = np.asarray(img) / np.iinfo(np.asarray(img).dtype).max

    logging.info(f"Deconvolving image")
    img = deconvolve(img, args.radius, args.sigma)

    if args.deconvolution_output:
        logging.info(f"Saving deconvolution adjustment file {args.deconvolution_output}")
        skimage.io.imsave(args.deconvolution_output, (img * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    img = np.power(img, args.root_power)
    if args.power_adjustment_output:
        logging.info(f"Saving power adjustment file {args.power_adjustment_output}")
        skimage.io.imsave(args.power_adjustment_output, (img * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    for i in range(args.sky_removal_passes):
        logging.info(f"Computing histogram")
        img[:,:,0] = histogramSmoothing(img[:,:,0].reshape(-1), args.hist_smoothing, args.sky_level_factor).reshape(img.shape[:-1])
        img[:,:,1] = histogramSmoothing(img[:,:,1].reshape(-1), args.hist_smoothing, args.sky_level_factor).reshape(img.shape[:-1])
        img[:,:,2] = histogramSmoothing(img[:,:,2].reshape(-1), args.hist_smoothing, args.sky_level_factor).reshape(img.shape[:-1])
        img = np.clip(img, 0, 1)

    if args.sky_removal_output:
        logging.info(f"Saving sky removal file {args.sky_removal_output}")
        skimage.io.imsave(args.sky_removal_output, (img * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    logging.info(f"Saving output file {args.output}")
    skimage.io.imsave(args.output, (img * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
