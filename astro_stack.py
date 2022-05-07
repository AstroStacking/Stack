#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import scipy as sp
from PIL import Image
import skimage
import sklearn.linear_model
import astropy.stats

def estimateLightPollution(img):
    """
    Estimate light pollution as a linear drop in the background
    """
    xv, yv = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='xy')
    X = np.array((xv.reshape(-1), yv.reshape(-1))).T
    y = img.reshape(-1, 3)
    reg = sklearn.linear_model.Lasso().fit(X, y)
    return reg.predict(X).reshape(img.shape)

def removeLightPollution(img):
    removed_light = img - estimateLightPollution(img)
    min = np.min(np.min(removed_light, axis=0), axis=0)
    return np.clip(removed_light - min, 0, 1)

def findStars(img):
    """
    Find stars on an image and return an array of possible stars
    """
    filtered_img = skimage.filters.gaussian(img, sigma=2, channel_axis=2)
    gray_img = skimage.color.rgb2gray(filtered_img)
    thresh_img = skimage.filters.threshold_otsu(gray_img)
    binary_img = gray_img > thresh_img
    labels = skimage.measure.label(binary_img, connectivity=2)
    properties = skimage.measure.regionprops(labels)
    return np.array([property.centroid[::-1] for property in properties])

def match_locations(img0, img1, coords0, coords1, radius=5, sigma=3, distance=10):
    """Match image locations using SSD minimization.

    Areas from `img0` are matched with areas from `img1`. These areas
    are defined as patches located around pixels with Gaussian
    weights.

    Parameters:
    -----------
    img0, img1 : 2D array
        Input images.
    coords0 : (2, m) array_like
        Centers of the reference patches in `img0`.
    coords1 : (2, n) array_like
        Centers of the candidate patches in `img1`.
    radius : int
        Radius of the considered patches.
    sigma : float
        Standard deviation of the Gaussian kernel centered over the patches.

    Returns:
    --------
    match_coords: (2, m) array
        The points in `coords1` that are the closest corresponding matches to
        those in `coords0` as determined by the (Gaussian weighted) sum of
        squared differences between patches surrounding each point.
    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    weights /= 2 * np.pi * sigma * sigma

    match_list = []
    for (c0, r0), coord0 in zip(coords0.astype(np.int16), coords0):
        roi0 = img0[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
        if roi0.shape != (2*radius+1, 2*radius+1):
            match_list.append((-1, -1))
            continue
        roi1_list = [(img1[r1 - radius:r1 + radius + 1,
                          c1 - radius:c1 + radius + 1], coords1) for (c1, r1), coords1 in zip(coords1.astype(np.int16), coords1)]

        # sum of squared differences
        ssd_list = [np.sum(weights * (roi0 - roi1[0]) ** 2) * np.exp(np.sum((coord0 - roi1[1])**2) / distance) if roi0.shape == roi1[0].shape else 10000000 for roi1 in roi1_list]
        match_list.append(coords1[np.argmin(ssd_list)])

    return np.array(match_list)

def findSkimageRegistration(stars, stars1_matched, shape):
    model_robust, inliers = skimage.measure.ransac((stars, stars1_matched), skimage.transform.SimilarityTransform, min_samples=30, residual_threshold=10, max_trials=100)
    """f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '"""
    logging.info(
      f"Translation: ({model_robust.translation[0]:.4f}, "
      f"{model_robust.translation[1]:.4f}), "
      f"Rotation: {model_robust.rotation:.4f}, "
      f"Inliers: {np.count_nonzero(inliers)}/{len(inliers)}"
      )
    return model_robust.inverse

def enhanceCoords(coords, shape):
    return np.column_stack([coords, coords[:,1]*np.cos(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.cos(np.pi/shape[1]*coords[:,1]), coords[:,1]*np.sin(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.sin(np.pi/shape[1]*coords[:,1])])

#def enhanceCoords(coords, shape):
#    return np.column_stack([coords, np.cos(np.pi/shape[0]*coords[:,0]), np.cos(np.pi/shape[1]*coords[:,1]), np.sin(np.pi/shape[0]*coords[:,0]), np.sin(np.pi/shape[1]*coords[:,1])])

#def enhanceCoords(coords, shape):
#    return coords

def findSkLearnRegistration(stars, stars1_matched, shape):
    model = sklearn.linear_model.RANSACRegressor(min_samples=30, residual_threshold=10)
    model.fit(enhanceCoords(stars, shape), stars1_matched)
    logging.info(
      f"Translation: ({model.estimator_.intercept_}), "
      f"Coeffs: {model.estimator_.coef_}, "
      f"Inliers: {np.count_nonzero(model.inlier_mask_)}/{len(model.inlier_mask_)}"
      )
    def predict(coords):
        return model.predict(enhanceCoords(coords, shape))
    return predict

#findRegistration = findSkimageRegistration
findRegistration = findSkLearnRegistration

def applyMultipleTransforms(transforms):
    """
    Create a function that will apply transforms in sequence
    """
    def apply(coords):
        for transform in transforms:
            coords = transform(coords)
        return coords
    return apply

def cleanImgs(imgs, filenames):
    cleaned_imgs = []
    for i in range(len(filenames)):
        logging.info(f"Cleaning up images {filenames[i]}")
        cleaned = removeLightPollution(imgs[i])
        cleaned_imgs.append(cleaned)
    return cleaned_imgs

def warpImgs(imgs, filenames):
    middle = len(filenames) // 2
    stars = []
    for i in range(len(filenames)):
        logging.info(f"Finding stars for {filenames[i]}")
        star = findStars(imgs[i])
        stars.append(star)

    grays = []
    for i in range(len(filenames)):
        logging.info(f"Greying for {filenames[i]}")
        gray = skimage.color.rgb2gray(imgs[i])
        grays.append(gray)
    
    models_forward = []
    for i in range(middle):
        logging.info(f"Registering {filenames[i]} to {filenames[i+1]}")
        stars_matched = match_locations(grays[i+1], grays[i], stars[i+1], stars[i])
        outlier_idxs = np.nonzero(stars_matched != 0)[0]
        model = findRegistration(stars[i+1], stars_matched, grays[middle].shape)
        models_forward.append([model])

    for i in range(1, len(models_forward)):
        models_forward[len(models_forward)-i-1].extend(models_forward[len(models_forward)-i])

    models_backward = []
    for i in range(middle+1, len(filenames)):
        logging.info(f"Registering {filenames[i]} to {filenames[i-1]}")
        stars_matched = match_locations(grays[i-1], grays[i], stars[i-1], stars[i])
        outlier_idxs = np.nonzero(stars_matched != 0)[0]
        model = findRegistration(stars[i-1], stars_matched, grays[middle].shape)
        models_backward.append([model])

    for i in range(1, len(models_backward)):
        models_backward[i].extend(models_backward[i-1])
        
    del grays
    del stars

    warps = []
    for i in range(0, len(models_forward)):
        logging.info(f"Applying {filenames[i]} registration")
        warp = skimage.transform.warp(imgs[i], applyMultipleTransforms(models_forward[i][::-1]))
        warps.append(warp)

    warps.append(imgs[middle])

    for i in range(0, len(models_backward)):
        logging.info(f"Applying {filenames[middle+i+1]} registration")
        warp = skimage.transform.warp(imgs[middle+i+1], applyMultipleTransforms(models_backward[i][::-1]))
        warps.append(warp)

    return warps

if __name__ == "__main__":
    import argparse
    import os
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Stack Astrophotography')
    parser.add_argument('images', type=str, nargs='+', help='images to stack')
    parser.add_argument('--unwrapped-output', type=str, help='Unwrapped image name')
    parser.add_argument('--pollution-output', type=str, help='Output image name')
    parser.add_argument('--registration-output', type=str, help='Output image name')
    parser.add_argument('--max-output', type=str, help='Output image name')
    parser.add_argument('--average-output', type=str, help='Output image name')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.add_argument('--no-register', dest='register', action='store_false')
    parser.set_defaults(register=True, clean=True, root_power=.6)

    args = parser.parse_args()
    filenames = args.images

    logging.info(f"Reading images {filenames}")
    imgs = [Image.open(filename) for filename in filenames]
    imgs = [np.asarray(img) / np.iinfo(np.asarray(img).dtype).max for img in imgs]

    if args.clean:
        imgs = cleanImgs(imgs, filenames)

        if args.pollution_output:
            os.makedirs(args.pollution_output, exist_ok=True)
            for i in range(len(filenames)):
                skimage.io.imsave(os.path.join(args.pollution_output, os.path.basename(filenames[i])), (imgs[i] * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    if args.register:
        if args.unwrapped_output:
            logging.info(f"Saving unwrapped file {args.unwrapped_output}")
            tmp_imgs = np.array(imgs)
            shape = tmp_imgs.shape[1:]
            unwrapped = np.max(tmp_imgs, axis=0).reshape(shape)
            skimage.io.imsave(args.unwrapped_output, (unwrapped * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
        imgs = warpImgs(imgs, filenames)

        if args.registration_output:
            for i in range(0, len(filenames)):
                os.makedirs(args.registration_output, exist_ok=True)
                skimage.io.imsave(os.path.join(args.registration_output, os.path.basename(filenames[i])), (imgs[i] * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    imgs = np.array(imgs)
    shape = imgs.shape[1:]
    if args.max_output:
        tmp_imgs = imgs.reshape(len(imgs), -1)
        logging.info(f"Computing max")
        max = np.max(imgs, axis=0).reshape(shape)
        logging.info(f"Saving max file {args.max_output}")
        skimage.io.imsave(args.max_output, (max * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

    if args.average_output:
        logging.info(f"Sigma clipping channel 1")
        average0 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,0].reshape(len(imgs), -1), axis=0)
        logging.info(f"Sigma clipping channel 2")
        average1 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,1].reshape(len(imgs), -1), axis=0)
        logging.info(f"Sigma clipping channel 3")
        average2 = astropy.stats.sigma_clipped_stats(imgs[:,:,:,2].reshape(len(imgs), -1), axis=0)
        average = np.column_stack((average0[1], average1[1], average2[1])).reshape(shape)
    
        logging.info(f"Saving pre stretch file {args.average_output}")
        skimage.io.imsave(args.average_output, (average * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')

