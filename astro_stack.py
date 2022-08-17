#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

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

class AstroStack:
    def __init__(self, args):
        import h5py

        self.args = args
        self.logger = logging.getLogger("AstroStack")
        
        if args.hdf5:
            self.hdf5 = h5py.File(args.hdf5, "w")
        else:
            import tempfile
            self.hdf5 = h5py.File(tempfile.TemporaryFile(), "w")
    
    def load_imgs(self):
        self.logger.info(f"Reading images {self.args.images}")
        imgs = []
        for i in trange(len(self.args.images), desc="Loading..."):
            img = Image.open(self.args.images[i])
            imgs.append(np.asarray(img) / np.iinfo(np.asarray(img).dtype).max)
        
        grp = self.hdf5.require_group("/inputs")
        data = grp.create_dataset("imgs", (len(imgs),) + imgs[0].shape)
        for i in trange(len(imgs), desc="HDF5..."):
            data[i] = imgs[i]
            
        return data
    
    def estimate_light_pollution(self, img):
        """
        Estimate light pollution as a linear drop in the background
        """
        xv, yv = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='xy')
        X = np.array((xv.reshape(-1), yv.reshape(-1))).T
        y = img.reshape(-1, 3)
        reg = sklearn.linear_model.Lasso().fit(X, y)
        return reg.predict(X).reshape(img.shape)

    def remove_light_pollution(self, img):
        removed_light = img - self.estimate_light_pollution(img)
        return np.clip(removed_light, 0, 1)

    def clean_imgs(self, imgs):
        if not self.args.clean:
            return imgs

        grp = self.hdf5.require_group("/clean")
        cleaned_imgs = grp.create_dataset("imgs", imgs.shape)
        for i in trange(len(self.args.images), desc="Cleaning..."):
            cleaned = self.remove_light_pollution(imgs[i])
            cleaned_imgs[i] = cleaned

        if self.args.pollution_output:
            os.makedirs(self.args.pollution_output, exist_ok=True)
            for i in range(len(self.args.images)):
                save(os.path.join(self.args.pollution_output, os.path.basename(self.args.images[i])), cleaned_imgs[i])
                
        return cleaned_imgs

    def find_stars(self, img, filename):
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

    def generate_potential_match(self, nbElements, fullRange):
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

    def match_graph(self, coords0, coords1):
        dist0 = sp.spatial.distance_matrix(coords0, coords0)
        dist1 = sp.spatial.distance_matrix(coords1, coords1)
        best = 1
        bestTrial = []
        
        for trial in self.generate_potential_match(len(coords0), len(coords1)):
            if len(set(trial)) != len(coords0):
                continue
            d = dist1[trial]
            d = d[:, trial]
            ratio = dist0 / d
            candidate = np.nanmax(np.abs(ratio-1))
            if candidate < best:
                best = candidate
                bestTrial = [(i, j) for i, j in enumerate(trial)]

        return bestTrial

    def save_match(self, img, stars, stars_ref, match, filename):
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.plot(stars[:, 0], stars[:, 1], "xr", markersize=4)
        plt.plot(stars_ref[:, 0], stars_ref[:, 1], "xb", markersize=4)
        for i in range(len(match[0])):
            plt.plot([match[0][i, 0], match[1][i, 0]], [match[0][i, 1], match[1][i, 1]], "-g")
        plt.savefig(f"{filename}.png")
        plt.close()
    
    def save_graph(self, img, stars, graph, stars_ref, graph_ref, filename):
        from scipy.spatial import KDTree
        interesting_stars = stars[graph]
        interesting_stars_ref = stars_ref[graph_ref]
        tree = KDTree(interesting_stars)
        tree_ref = KDTree(interesting_stars_ref)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.plot(stars[:, 0], stars[:, 1], "xr", markersize=4)
        for i in range(len(graph)):
            data, indices = tree.query(interesting_stars[i], k=3)
            for index in indices:
                plt.plot([interesting_stars[i, 0], interesting_stars[index, 0]], [interesting_stars[i, 1], interesting_stars[index, 1]], "-b")
        for i in range(len(graph_ref)):
            data, indices = tree_ref.query(interesting_stars_ref[i], k=3)
            for index in indices:
                plt.plot([interesting_stars_ref[i, 0], interesting_stars_ref[index, 0]], [interesting_stars_ref[i, 1], interesting_stars_ref[index, 1]], "-g")
        plt.savefig(f"graph_{filename}.png")
        plt.close()

    def find_full_graph_match(self, coords0, coords1, minFullGraph):
        for i in range(minFullGraph):
            mainGraph = self.match_graph(coords0[i:i+minFullGraph], coords1[:minFullGraph*2])
            if len(mainGraph) != 0:
                break
        return [i for i, j in mainGraph], [j for i, j in mainGraph]

    def find_partial_graph_match(self, matchList, coords0, coords1):
        matchSet = set(matchList[1])

        fulldist0 = sp.spatial.distance_matrix(coords0, coords0)
        fulldist1 = sp.spatial.distance_matrix(coords1, coords1)

        for i in trange(max(matchList[0]), len(coords0), desc="Partial graph match"):
            best = 1
            bestTrial = []
            
            dist0 = fulldist0[i, matchList[0]]
            for j in range(0, len(coords1)):
                if j in matchSet:
                    continue
                dist1 = fulldist1[j, matchList[1]]
                ratio = dist0/dist1
                candidate = np.nanmax(np.abs(ratio-1))
                if candidate < best:
                    best = candidate
                    bestTrial = (i, j)
            
            if best < 0.01:
                matchList[0].append(bestTrial[0])
                matchList[1].append(bestTrial[1])
                matchSet.add(bestTrial[1])

        return matchList

    def match_locations(self, img, coords0, coords1, minFullGraph, filename):
        matchList = self.find_full_graph_match(coords0, coords1, minFullGraph)
        matchList = self.find_partial_graph_match(matchList, coords0, coords1)

        self.save_graph(img, coords1, matchList[1], coords0, matchList[0], filename)
        return np.array([[coords0[m] for m in matchList[0]], [coords1[m] for m in matchList[1]]])

    def find_skimage_registration(self, stars, stars1_matched, shape, enhance_coords):
        model_robust, inliers = skimage.measure.ransac((stars, stars1_matched), skimage.transform.SimilarityTransform, min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=100)
        return model_robust.inverse

    def enhance_coords_full(self, coords, shape):
        return np.column_stack([coords, coords[:,1]*np.cos(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.cos(np.pi/shape[1]*coords[:,1]), coords[:,1]*np.sin(np.pi/shape[0]*coords[:,0]), coords[:,0]*np.sin(np.pi/shape[1]*coords[:,1])])

    def enhance_coords_partial(self, coords, shape):
        return np.column_stack([coords, np.cos(np.pi/shape[0]*coords[:,0]), np.cos(np.pi/shape[1]*coords[:,1]), np.sin(np.pi/shape[0]*coords[:,0]), np.sin(np.pi/shape[1]*coords[:,1])])

    def enhance_coords_direct(self, coords, shape):
        return coords

    def find_sklearn_registration(self, stars, stars1_matched, shape, enhance_coords):
        model = sklearn.linear_model.RANSACRegressor(min_samples=max(len(stars)//2, 10), residual_threshold=10, max_trials=1000)
        model.fit(enhance_coords(stars, shape), stars1_matched)
        def predict(coords):
            return model.predict(enhance_coords(coords, shape))
        return predict

    #find_registration = find_skimage_registration
    find_registration = find_sklearn_registration

    def warp_imgs(self, imgs):
        middle = len(self.args.images) // 2
        
        enhance_coords = self.enhance_coords_partial

        register = self.hdf5.require_group("/register")
        stars = self.hdf5.require_group("/register/stars")
        graphs = self.hdf5.require_group("/register/graphs")
        grays = self.hdf5.require_group("/register/grays")
        
        for i in trange(len(self.args.images), desc="Finding stars"):
            star = self.find_stars(imgs[i], self.args.images[i])
            stars.create_dataset(self.args.images[i], star.shape, dtype=star.dtype)[:] = star
            gray = skimage.color.rgb2gray(imgs[i])
            grays.create_dataset(self.args.images[i], gray.shape, dtype=gray.dtype)[:] = gray
        
        shape = grays[self.args.images[middle]].shape
        warps = register.create_dataset("imgs", imgs.shape)
        for i in trange(middle, desc="Registering forward to middle"):
            stars_matched = self.match_locations(imgs[i], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]), self.args.full_graph, self.args.images[i])
            if stars_matched.shape[1] < 10:
                stars_matched = self.match_locations(imgs[i], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]), self.args.full_graph-1, self.args.images[i])
            graphs.create_dataset(self.args.images[i], stars_matched.shape, dtype=stars_matched.dtype)[:] = stars_matched
            model = self.find_registration(stars_matched[0], stars_matched[1], shape, enhance_coords)
            warps[i] = skimage.transform.warp(imgs[i], model)

        warps[middle] = imgs[middle]

        for i in trange(middle+1, len(self.args.images), desc="Registering backward to middle"):
            stars_matched = self.match_locations(imgs[i], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]), self.args.full_graph, self.args.images[i])
            if stars_matched.shape[1] < 10:
                stars_matched = self.match_locations(imgs[i], np.array(stars[self.args.images[middle]]), np.array(stars[self.args.images[i]]), self.args.full_graph-1, self.args.images[i])
            graphs.create_dataset(self.args.images[i], stars_matched.shape, dtype=stars_matched.dtype)[:] = stars_matched
            model = self.find_registration(stars_matched[0], stars_matched[1], shape, enhance_coords)
            warps[i] = skimage.transform.warp(imgs[i], model)

        return warps

    def register_imgs(self, imgs):
        if self.args.unwrapped_output:
            self.logger.info(f"Saving unwrapped file {args.unwrapped_output}")
            unwrapped = np.max(imgs, axis=0)
            save(self.args.unwrapped_output, unwrapped)
        imgs = self.warp_imgs(imgs)

        if args.registration_output:
            for i in range(0, len(self.args.images)):
                os.makedirs(self.args.registration_output, exist_ok=True)
                save(os.path.join(self.args.registration_output, os.path.basename(self.args.images[i])), imgs[i])
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
            save(self.args.max_output, max)

        if self.args.average_output:
            data = []
            for i in trange(imgs.shape[1], desc="Sigma clipping channel"):
                data.append(astropy.stats.sigma_clipped_stats(imgs[:,i,:,:].reshape(len(imgs), -1), axis=0)[0])
            average = np.column_stack(data).reshape(imgs.shape[1:])
        
            self.logger.info(f"Saving pre stretch file {args.average_output}")
            save(self.args.average_output, average)

        self.hdf5.close()

if __name__ == "__main__":
    import argparse
    import logging
    import os
    
    np.seterr(divide='ignore', invalid='ignore')

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
    parser.add_argument('--full-graph', type=int, default=5)

    parser.add_argument('--max-output', type=str, help='Output image name')
    parser.add_argument('--average-output', type=str, help='Output image name')

    args = parser.parse_args()
    
    stack = AstroStack(args)
    stack()

