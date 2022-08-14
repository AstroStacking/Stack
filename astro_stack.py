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

MAX_STARS = 100
MIN_STARS = 80

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
    return np.clip(removed_light, 0, 1)

def findStars(img, filename):
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
        if len(properties) > MAX_STARS:
            thresh_fix = thresh_fix * 1.1
        elif len(properties) < MIN_STARS:
            thresh_fix = thresh_fix * .95
        else:
            break
    properties.sort(key=lambda property: property.area, reverse=True)
    data = np.array([property.centroid[::-1] for property in properties])
    skimage.io.imsave(f"thres_{filename}", (binary_img * np.iinfo(np.uint16).max).astype(np.uint16), plugin='tifffile')
    np.savetxt(f"thres_{filename}.txt", data, delimiter=",")
    return data

def generatePotentialMatch(nbElements, fullRange):
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

def matchGraph(coords0, coords1):
    dist0 = sp.spatial.distance_matrix(coords0, coords0)
    dist1 = sp.spatial.distance_matrix(coords1, coords1)
    
    for trial in generatePotentialMatch(len(coords0), len(coords1)):
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

def matchLocations(img0, img1, coords0, coords1, minFullGraph=5):
    
    mainGraph = matchGraph(coords0[:minFullGraph], coords1[:minFullGraph*2])
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

def saveMatch(img, stars, stars_ref, match, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.plot(stars[:, 0], stars[:, 1], "xr", markersize=4)
    plt.plot(stars_ref[:, 0], stars_ref[:, 1], "xb", markersize=4)
    for i in range(len(match[0])):
        plt.plot([match[0][i, 0], match[1][i, 0]], [match[0][i, 1], match[1][i, 1]], "-g")
    plt.savefig(f"{filename}.png")
    plt.close()

def findSkimageRegistration(stars, stars1_matched, shape):
    model_robust, inliers = skimage.measure.ransac((stars, stars1_matched), skimage.transform.SimilarityTransform, min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=100)
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

def enhanceCoords(coords, shape):
    return coords

def findSkLearnRegistration(stars, stars1_matched, shape):
    model = sklearn.linear_model.RANSACRegressor(min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=1000)
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
        logging.info(f"Cleaning up image {filenames[i]}")
        cleaned = removeLightPollution(imgs[i])
        cleaned_imgs.append(cleaned)
    return cleaned_imgs

def warpImgs(imgs, filenames):
    middle = len(filenames) // 2
    stars = []
    for i in range(len(filenames)):
        logging.info(f"Finding stars for {filenames[i]}")
        star = findStars(imgs[i], filenames[i])
        logging.info(f"Found: {len(star)}")
        stars.append(star)

    grays = []
    for i in range(len(filenames)):
        logging.info(f"Greying for {filenames[i]}")
        gray = skimage.color.rgb2gray(imgs[i])
        grays.append(gray)
    
    models_forward = []
    for i in range(middle):
        logging.info(f"Registering {filenames[i]} to {filenames[i+1]}")
        stars_matched = matchLocations(grays[i+1], grays[i], stars[i+1], stars[i])
        saveMatch(imgs[i], stars[i+1], stars[i], stars_matched, filenames[i])
        model = findRegistration(stars_matched[0], stars_matched[1], grays[middle].shape)
        models_forward.append([model])

    for i in range(1, len(models_forward)):
        models_forward[len(models_forward)-i-1].extend(models_forward[len(models_forward)-i])

    models_backward = []
    for i in range(middle+1, len(filenames)):
        logging.info(f"Registering {filenames[i]} to {filenames[i-1]}")
        stars_matched = matchLocations(grays[i-1], grays[i], stars[i-1], stars[i])
        saveMatch(imgs[i], stars[i-1], stars[i], stars_matched, filenames[i])
        model = findRegistration(stars_matched[0], stars_matched[1], grays[middle].shape)
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

