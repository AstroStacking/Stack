#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math

import numpy as np
import scipy as sp
from PIL import Image
import skimage
import sklearn.linear_model
import astropy.stats
from matplotlib import pyplot as plt
import h5py

class AstroStack:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("AstroStack")
        
        if args.hdf5:
            self.hdf5 = h5py.File(args.hdf5, "w")
        else:
            import tempfile
            self.hdf5 = h5py.File(tempfile.TemporaryFile(), "w")
    
    def load_imgs(self):
        self.logger.info(f"Reading images {self.args.images}")
        imgs = [Image.open(filename) for filename in self.args.images]
        imgs = [np.asarray(img) / np.iinfo(np.asarray(img).dtype).max for img in imgs]
        
        grp = self.hdf5.require_group("/inputs")
        data = grp.create_dataset("imgs", (len(imgs),) + imgs[0].shape)
        for i in range(len(imgs)):
            data[i] = imgs[i]
            
        return data
    
    def estimateLightPollution(self, img):
        """
        Estimate light pollution as a linear drop in the background
        """
        xv, yv = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='xy')
        X = np.array((xv.reshape(-1), yv.reshape(-1))).T
        y = img.reshape(-1, 3)
        reg = sklearn.linear_model.Lasso().fit(X, y)
        return reg.predict(X).reshape(img.shape)

    def removeLightPollution(self, img):
        removed_light = img - self.estimateLightPollution(img)
        return np.clip(removed_light, 0, 1)

    def clean_imgs(imgs):
        if not self.args.clean:
            return imgs

        grp = self.hdf5.require_group("/clean")
        cleaned_imgs = grp.create_dataset("imgs", imgs.shape)
        for i in range(len(self.args.images)):
            self.logger.info(f"Cleaning up image {self.args.images[i]}")
            cleaned = self.removeLightPollution(imgs[i])
            cleaned_imgs[i] = cleaned

        if self.args.pollution_output:
            os.makedirs(self.args.pollution_output, exist_ok=True)
            for i in range(len(self.args.images)):
                skimage.io.imsave(os.path.join(self.args.pollution_output, os.path.basename(self.args.images[i])), (cleaned_imgs[i] * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
                
        return cleaned_imgs

    def findStars(self, img, filename):
        """
        Find stars on an image and return an array of possible stars
        """
        gray_img = skimage.color.rgb2gray(img)
        filtered_img = skimage.filters.gaussian(gray_img, sigma=10)
        thresh_fix = 1
        while True:
            thresh = skimage.filters.threshold_otsu(filtered_img)
            binary_img = filtered_img > thresh * thresh_fix
            labels = skimage.measure.label(binary_img, connectivity=2)
            properties = skimage.measure.regionprops(labels)
            if len(properties) > self.args.max_stars:
                thresh_fix = thresh_fix * 1.1
            elif len(properties) < self.args.min_stars:
                thresh_fix = thresh_fix * .95
            else:
                break
        properties.sort(key=lambda property: property.area, reverse=True)
        data = np.array([property.centroid[::-1] for property in properties if property.axis_minor_length > 0 and property.axis_major_length / property.axis_minor_length < 5])
        return data

    def generatePotentialMatch(self, nbElements, fullRange):
        l = [0] * nbElements
        while True:
            inRange = False
            for i in range(nbElements):
                l[i] = l[i] + 1
                if l[i] == fullRange:
                    l[i] = 0
                else:
                    inRange = True
                    break
            if inRange:
                yield l
            else:
                break

    def matchGraph(self, coords0, coords1):
        dist0 = sp.spatial.distance_matrix(coords0, coords0)
        dist1 = sp.spatial.distance_matrix(coords1, coords1)
        
        for trial in self.generatePotentialMatch(len(coords0), len(coords1)):
            if len(set(trial)) != len(coords0):
                continue
            d = dist1[trial]
            d = d[:, trial]
            ratio = dist0 / d
            ok = True
            for i in range(1, len(coords0)):
                for j in range(i+1, len(coords0)):
                    if ratio[i, j] > 1.1 or ratio[i, j] < 0.9:
                        ok = False
            if ok:
                return [(i, j) for i, j in enumerate(trial)]

        return []

    def matchLocations(self, img0, img1, coords0, coords1, minFullGraph=5):

        for i in range(minFullGraph):
            mainGraph = self.matchGraph(coords0[i:i+minFullGraph], coords1[:minFullGraph*2])
            if len(mainGraph) != 0:
                break
        matchList = [i for i, j in mainGraph], [j for i, j in mainGraph]
        
        matchSet = set(matchList[1])

        for i in range(minFullGraph, len(coords0)):
            dist0 = sp.spatial.distance_matrix(coords0[i, :][None, :], coords0[matchList[0]])
            for j in range(0, len(coords1)):
                if j in matchSet:
                    continue
                dist1 = sp.spatial.distance_matrix(coords1[j, :][None, :], coords1[matchList[1]])
                ratio = dist0/dist1
                if np.all(ratio < 1.1) and np.all(ratio > 0.9):
                    matchList[0].append(i)
                    matchList[1].append(j)
                    matchSet.add(j)
                    break

        return np.array([[coords0[m] for m in matchList[0]], [coords1[m] for m in matchList[1]]])

    def findSkimageRegistration(self, stars, stars1_matched, shape):
        model_robust, inliers = skimage.measure.ransac((stars, stars1_matched), skimage.transform.SimilarityTransform, min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=100)
        """f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '"""
        self.logger.info(
          f"Translation: ({model_robust.translation[0]:.4f}, "
          f"{model_robust.translation[1]:.4f}), "
          f"Rotation: {model_robust.rotation:.4f}, "
          f"Inliers: {np.count_nonzero(inliers)}/{len(inliers)}"
          )
        return model_robust.inverse

    def enhanceCoords(self, coords, shape):
        return np.column_stack([coords, coords[:,1]*np.cos(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.cos(np.pi/shape[1]*coords[:,1]), coords[:,1]*np.sin(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.sin(np.pi/shape[1]*coords[:,1])])

    #def enhanceCoords(self, coords, shape):
    #    return np.column_stack([coords, np.cos(np.pi/shape[0]*coords[:,0]), np.cos(np.pi/shape[1]*coords[:,1]), np.sin(np.pi/shape[0]*coords[:,0]), np.sin(np.pi/shape[1]*coords[:,1])])

    def enhanceCoords(self, coords, shape):
        return coords

    def findSkLearnRegistration(self, stars, stars1_matched, shape):
        model = sklearn.linear_model.RANSACRegressor(min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=1000)
        model.fit(self.enhanceCoords(stars, shape), stars1_matched)
        self.logger.info(
          f"Translation: ({model.estimator_.intercept_}), "
          f"Coeffs: {model.estimator_.coef_}, "
          f"Inliers: {np.count_nonzero(model.inlier_mask_)}/{len(model.inlier_mask_)}"
          )
        def predict(coords):
            return model.predict(self.enhanceCoords(coords, shape))
        return predict

    #findRegistration = findSkimageRegistration
    findRegistration = findSkLearnRegistration

    def warpImgs(self, imgs):
        middle = len(self.args.images) // 2

        register = self.hdf5.require_group("/register")
        stars = self.hdf5.require_group("/register/stars")
        graphs = self.hdf5.require_group("/register/graphs")
        grays = self.hdf5.require_group("/register/grays")
        for i in range(len(self.args.images)):
            self.logger.info(f"Finding stars for {self.args.images[i]}")
            star = self.findStars(imgs[i], self.args.images[i])
            stars.create_dataset(self.args.images[i], star.shape, dtype=star.dtype)[:] = star
            self.logger.info(f"Found: {len(star)}")
            gray = skimage.color.rgb2gray(imgs[i])
            grays.create_dataset(self.args.images[i], gray.shape, dtype=gray.dtype)[:] = gray
        
        shape = grays[self.args.images[middle]].shape
        
        models_forward = []
        for i in range(middle):
            self.logger.info(f"Registering {self.args.images[i]} to {self.args.images[middle]}")
            stars_matched = self.matchLocations(grays[self.args.images[middle]], grays[self.args.images[i]], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]))
            graphs.create_dataset(self.args.images[i], stars_matched.shape, dtype=stars_matched.dtype)[:] = stars_matched
            model = self.findRegistration(stars_matched[0], stars_matched[1], shape)
            models_forward.append(model)

        models_backward = []
        for i in range(middle+1, len(self.args.images)):
            self.logger.info(f"Registering {self.args.images[i]} to {self.args.images[middle]}")
            stars_matched = self.matchLocations(grays[self.args.images[middle]], grays[self.args.images[i]], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]))
            graphs.create_dataset(self.args.images[i], stars_matched.shape, dtype=stars_matched.dtype)[:] = stars_matched
            model = self.findRegistration(stars_matched[0], stars_matched[1], shape)
            models_backward.append(model)

        warps = register.create_dataset("imgs", imgs.shape)
        for i in range(0, len(models_forward)):
            self.logger.info(f"Applying {self.args.images[i]} registration")
            warp = skimage.transform.warp(imgs[i], models_forward[i])
            warps[i] = warp

        warps[middle] = imgs[middle]

        for i in range(0, len(models_backward)):
            self.logger.info(f"Applying {self.args.images[middle+i+1]} registration")
            warp = skimage.transform.warp(imgs[middle+i+1], models_backward[i])
            warps[middle + i] = warp

        return warps

    def register_imgs(self, imgs):
        if self.args.unwrapped_output:
            self.logger.info(f"Saving unwrapped file {args.unwrapped_output}")
            unwrapped = np.max(imgs, axis=0)
            skimage.io.imsave(self.args.unwrapped_output, (unwrapped * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
        imgs = self.warpImgs(imgs)

        if args.registration_output:
            for i in range(0, len(self.args.images)):
                os.makedirs(self.args.registration_output, exist_ok=True)
                skimage.io.imsave(os.path.join(self.args.registration_output, os.path.basename(self.args.images[i])), (imgs[i] * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
        return imgs

    def __call__(self):
        imgs = self.load_imgs()
        if self.args.clean:
            imgs = self.clean_imgs(imgs)
        
        if self.args.register:
            imgs = self.register_imgs(imgs)

        if self.args.max_output:
            self.logger.info(f"Computing max")
            max = np.max(imgs, axis=0)
            self.logger.info(f"Saving max file {args.max_output}")
            skimage.io.imsave(self.args.max_output, (max * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

        if self.args.average_output:
            self.logger.info(f"Sigma clipping channel 1")
            average0 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,0].reshape(len(imgs), -1), axis=0)
            self.logger.info(f"Sigma clipping channel 2")
            average1 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,1].reshape(len(imgs), -1), axis=0)
            self.logger.info(f"Sigma clipping channel 3")
            average2 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,2].reshape(len(imgs), -1), axis=0)
            average = np.column_stack((average0[1], average1[1], average2[1])).reshape(imgs.shape[1:])
        
            self.logger.info(f"Saving pre stretch file {args.average_output}")
            skimage.io.imsave(self.args.average_output, (average * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

        self.hdf5.close()

if __name__ == "__main__":
    import argparse
    import os
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Stack Astrophotography')
    parser.add_argument('images', type=str, nargs='+', help='images to stack')

    parser.add_argument('--hdf5', type=str, help='HDF5 file name')
    parser.add_argument('--unwrapped-output', type=str, help='Unwrapped image name')

    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--pollution-output', type=str, help='Output image name')

    parser.add_argument('--registration-output', type=str, help='Output image name')
    parser.add_argument('--no-register', dest='register', action='store_false')
    parser.add_argument('--min-stars', type=int, default=80)
    parser.add_argument('--max-stars', type=int, default=100)

    parser.add_argument('--max-output', type=str, help='Output image name')
    parser.add_argument('--average-output', type=str, help='Output image name')

    args = parser.parse_args()
    
    stack = AstroStack(args)
    stack()

