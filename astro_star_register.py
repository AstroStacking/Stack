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

def transform(coords, focal):
    #return np.tan(coords/ focal)
    return coords

def predict_all(coords, angle, zoom, origin, focal=10000.):
    return (np.dot(transform(coords, focal) - origin, np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))) + origin) * zoom

def cost_all(graph0, graph1, angle, zoom, origin, focal=10000.):
    return np.sum((predict_all(transform(graph0, focal), angle, zoom, origin) - transform(graph1, focal)) ** 2)

def partial_gradient_angle(coords, angle, zoom, origin, focal=10000.):
    return np.dot(transform(coords, focal) - origin, np.array(((-math.sin(angle), math.cos(angle)), (-math.cos(angle), -math.sin(angle))))) * zoom

def partial_gradient_zoom(coords, angle, zoom, origin, focal=10000.):
    return np.dot(transform(coords, focal) - origin, np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))) + origin
    
def partial_gradient_origin(coords, angle, zoom, origin, focal=10000.):
    return np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))

class AstroRegister:
    def __init__(self, args):
        import h5py

        self.args = args
        self.logger = logging.getLogger("AstroRegister")
        
        self.hdf5 = h5py.File(args.hdf5, "r")
        self.delta = (2736, 1824)

    def find_direct_registration(self, stars, stars1_matched, shape, enhance_coords):
        model = sklearn.linear_model.LinearRegression()
        model.fit(enhance_coords(stars, shape), stars1_matched)
        def predict(coords):
            return model.predict(enhance_coords(coords, shape))
        return predict
        
    def optimise(self, graph):
        graph0 = graph[0] - self.delta
        graph1 = graph[1] - self.delta

        def cost(X):
            return cost_all(graph0, graph1, X[0], X[1], np.array((X[2], X[3])))
        def gradient(X):
            angles = partial_gradient_angle(graph0, X[0], X[1], np.array((X[2], X[3])))
            zooms = partial_gradient_zoom(graph0, X[0], X[1], np.array((X[2], X[3])))
            origins = partial_gradient_origin(graph0, X[0], X[1], np.array((X[2], X[3])))
            
            partial_gradient = np.zeros((4, *angles.shape))
            partial_gradient[0] = angles
            partial_gradient[1] = zooms
            partial_gradient[2:] = -X[1] * origins[:, None, :]
            partial_gradient[2,:,0] += X[1]
            partial_gradient[3,:,1] += X[1]
            grad = 2 * np.dot(partial_gradient.reshape(4, -1), (predict_all(graph0, X[0], X[1], np.array((X[2], X[3]))) - graph1).reshape(-1))
            
            return grad

        from scipy.optimize import minimize
        
        r = minimize(cost, (0, 1, 0, 0), jac=gradient)
        print(r.fun)
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

