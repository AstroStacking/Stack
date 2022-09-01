#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import itertools

import numpy as np
import scipy as sp
from PIL import Image
import skimage
import sklearn.linear_model
import astropy.stats
from matplotlib import pyplot as plt
from tqdm.auto import trange

def save(filename, data):
    skimage.io.imsave(filename, (data * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile', check_contrast=False)

def predict_all(coords, angle, zoom, origin):
    return (np.dot(coords - origin, np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))) + origin) * zoom

class AstroRegister:
    def __init__(self, args):
        import h5py

        self.args = args
        self.logger = logging.getLogger("AstroRegister")
        
        self.hdf5 = h5py.File(args.hdf5, "r")

    def find_direct_registration(self, stars, stars1_matched, shape, enhance_coords):
        model = sklearn.linear_model.LinearRegression()
        model.fit(enhance_coords(stars, shape), stars1_matched)
        def predict(coords):
            return model.predict(enhance_coords(coords, shape))
        return predict
    
    def cost(self, graph0, graph1, angle, zoom, origin):
        return np.sum((graph1 - predict_all(graph0, angle, zoom, origin)) ** 2)
    
    def optimise(self, graph):
        graph0 = graph[0]
        graph1 = graph[1]
        
        def cost(X):
            return self.cost(graph0, graph1, X[0], X[1], (X[2], X[3]))
        
        from scipy.optimize import minimize
        
        r = minimize(cost, (0, 1, 0, 0))
        print(r.x)

    def __call__(self):
        for graph in self.hdf5["register"]["graphs"].values():
            self.optimise(graph[:])

        self.hdf5.close()

if __name__ == "__main__":
    import argparse
    import logging
    import os
    
    np.seterr(divide='ignore', invalid='ignore')

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Stack Astrophotography')

    parser.add_argument('--hdf5', type=str, help='HDF5 file name')
    parser.add_argument('--compensate-barillet', dest='barillet', default='none', choices=['none', 'partial', 'full'], help='Tries to compensate for barillet distortion')
    parser.add_argument('--optimise', default='ransac', choices=['ransac', 'direct'], help='Optimisation procedure')

    args = parser.parse_args()
    
    register = AstroRegister(args)
    register()

