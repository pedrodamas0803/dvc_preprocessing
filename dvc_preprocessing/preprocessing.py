#!/usr/bin/env python
# coding: utf-8

import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import h5py
import skimage.io as skio
from skimage import exposure, filters
# from skimage.util.dtype import img_as_int, img_as_ubyte
from skimage.measure import regionprops
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from numpy import linalg as LA
import os
import copy
from . import plot as prplot

"""

Routine to preprocess PCT reconstructions to perform DVC analysis

"""


def read_images_from_h5(filename, data_type = 'int16', dirpath="./"):
    """
    Returns the stack image of the reconstructed volume
    Ideally you should provide the hdf5 file provided after nabu volume reconstruction for data entry consistency.
    Inputs

    filename - string with the h5 file name 
    data_type - data type from the returned stack image, whether 'int8', 'int16', or 'original'. The latter will import the original type without casting
    dirpath - the directory where the h5 named 'filename' is found

    """
    if data_type not in ['int8', 'int16', 'original']:
        raise TypeError("Your image won't be suitable for DVC analysis. Choose between data_type = int8 or int16.")

    with h5py.File(os.path.join(dirpath, filename)) as h5:
        stack_img = h5['entry0000']['reconstruction']['results']['data'][:]

    if data_type == 'int8':

        return stack_img.astype(np.int8)
        
    elif data_type == 'original':
        
        return stack_img
        
    else:

        return stack_img.astype(np.int16)

# def convert_stack(image, data_type = 'int16'):

#     """
#     Converts the data type from the original reconstruction. By default nabu spits out images in np.int32.
#     """

#     if data_type not in ['int8', 'int16']:
#         raise TypeError("Your image won't be suitable for DVC")

#     elif data_type == 'int8':
#         return img_as_ubyte(image)

#     return img_as_int(image)



def crop_slice(image, slice_number, xmin, xmax, ymin, ymax):
    '''
    Returns the cropped slices according to the chosen parameters

    Inputs

    image - stack image from which the slice will be taken
    slice_number - well, you know...
    xmin, xmax, ymin, ymax - maximum and minimum values for each axis 
    '''

    return image[slice_number][ymin:ymax, xmin:xmax]


def crop_stack(image, xmin, xmax, ymin, ymax):
    '''
    Returns the cropped stack image, i.e.: crops all slices according to the chosen coordinates

    Inputs

    image - stack image from which the slice will be taken
    slice_number - well, you know...
    xmin, xmax, ymin, ymax - maximum and minimum values for each axis
    '''

    return image[:, ymin:ymax, xmin:xmax]


def intensity_rescaling(image, low_perc=1, high_perc=99):
    '''
    Rescales the intensity of the stack image according to the chosen percentiles 

    Inputs

    image - stack image to be rescaled
    low_perc, high_perc - low and high percentiles to use as parameters to rescale the images
    '''

    plow, phigh = np.percentile(image, (low_perc, high_perc))

    rescaled_image = exposure.rescale_intensity(image, in_range=(plow, phigh))

    return rescaled_image


def _find_center_of_mass(image:np.ndarray):
    '''
    Finds the center of mass of the image after applying Otsu's thresholding algorithm

    Input
    image - usually a slice of a reconstructed volume

    Output
    coordinates of the weighted center of mass in the form Y, X for consistency with matplotlib.pyplot

    '''
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    weighted_center_of_mass = properties[0].weighted_centroid

    return weighted_center_of_mass


def volume_CoM(image:np.ndarray, slab_size:int = 600):
    '''
    Calculates the average coordinates of the center of mass calculated over the range init_slice to final_slice

    Inputs

    image - the stack image that will be iterated over
    init_slice - starting slice for computation of the CoM. default = 0
    final_slice - final slice for computation of CoM. default = len(image)

    Output
    coordinates of the weighted center of mass in the form Y, X for consistency with matplotlib.pyplot

    '''
    nz, ny, nx = image.shape

    vol = image[(nz//2 - slab_size//2):(nz//2+slab_size//2)]
    print(vol.shape)
    x = np.zeros(slab_size)
    y = np.zeros(slab_size)

    with concurrent.futures.ProcessPoolExecutor() as pool:

        for ii, result in enumerate(pool.map(_find_center_of_mass, vol)):

            y[ii], x[ii] = result

    return np.mean(y), np.mean(x)


def crop_around_CoM(image:np.ndarray, CoM: tuple, zprop:float = 1, xprop:float = 0.25, yprop:float = 0.25):
    '''
    This function will return the image of an slice cut with parameters relative to the calculated center of mass

    Inputs
    image - the image!
    CoM - coordinates of the center of mass as a tuple
    slices - tuple containing the starting 
    '''
    try:
        zlen, ylen, xlen = image.shape
    except ValueError:
        ylen, xlen = image.shape
    
    
    ycom, xcom = CoM
    zcom = zlen // 2

    xmin = int(np.ceil(xcom - (xlen * xprop)))
    xmax = int(np.floor(xcom + (xlen * xprop)))
    ymin = int(np.ceil(ycom - (ylen * yprop)))
    ymax = int(np.floor(ycom + (ylen * yprop)))
    zmin = int(np.ceil(zcom - (zlen * zprop)))
    zmax = int(np.floor(zcom + (zlen * zprop)))

    if xmin < 0:
        xmin = 0
    if xmax > xlen:
        xmax = xlen
    
    if ymin < 0:
        ymin = 0
    if ymax > ylen:
        ymax = ylen
    
    if zmin < 0:
        zmin=0
    if zmax > zlen:
        zmax = zlen

    return image[zmin:zmax, ymin:ymax, xmin:xmax]


def get_rotation_angle(image, plot=False, canny_sigma=30, hough_thrs=5, line_len=150, line_gap=10, mean = True):
    
    '''
    This function gets the rotation angle from the image averaged along the Z axis (axis=0 in np).
    Inputs
    
    image - the stack image
    plot - False by default. If True plots the image together with the detected edges and the calculated lines.
    canny_sigma - the standard deviation for the Gaussian filter used in the Canny edge detector.
    hough_thrs - threshold for the determination of the lines by probabilistic line Hough transform.
    line_len - minimum length accepted for the detected lines.
    line_gap - maximum gap between two pixels to still be considered a line.
    mean - True by default. If set to False it will return the array of calculated angles from the image. True will return the mean of the values. 
    '''

    # Line finding using the Probabilistic Hough Transform
    img = copy.deepcopy(image)
    if img.ndim == 3:
        img = img.mean(axis=0)
    edges = canny(img, canny_sigma)
    x, y = edges.shape
    edges = edges[int(0.05*x):int(0.95*x), int(0.05*y):int(0.95*y)]
    img = img[int(0.05*x):int(0.95*x), int(0.05*y):int(0.95*y)]
    lines = probabilistic_hough_line(edges, hough_thrs, line_len,
                                     line_gap, seed =2)
    prob_angles = []

    for line in lines:
        y = line[1][1]-line[0][1]
        x = line[1][0]-line[0][0]
        vec = x, y
        vec_norm = LA.norm(vec)
        # print(vec, vec_norm)

        if vec[0] != 0 and vec[1] != 0:
            ang = (np.rad2deg(np.arctan(y/x)))
            prob_angles.append(ang)

    corr_ang = []
    for angle in prob_angles:
        if angle < 0:
            corr_ang.append(angle+90)
        else:
            corr_ang.append(angle)
            
    if plot == True:
        prplot.plot_angle_detection(img, edges, lines)

    if not mean:
        return corr_ang
    else:
        return np.mean(np.mean(corr_ang))


def save_3d_tiff(image, filename="output", path="./"):
    '''
    Saves a 3D tiff image suitable to perform DVC analysis

    Inputs

    image - stack image to be saved
    filename - 'output' by default, which is a pretty useless name
    path - './' by default
    '''
    x, y, z = image.shape
    image_3d = skio.concatenate_images(image)
    skio.imsave(os.path.join(
        path, f"{filename}_{x}_{y}_{z}.tiff"), arr=image_3d, plugin="tifffile")


def save_3d_subset_tiff(image, init_slice, end_slice, filename, path="./"):
    '''
    Saves a subset of slices in a tomo result.

    Inputs

    image - stack image to be sliced and saved
    init_slice - first slice of the stack to be saved as tiff
    end_slice - last slice of the stack to be saved as tiff
    filename - stem for the filename that will be added with the shape of the image
    path - path to the directory where you want to save tour image
    '''

    x, y, z = image.shape
    image_3d = skio.concatenate_images(image[init_slice:end_slice])
    skio.imsave(os.path.join(
        path, f"{filename}_{x-init_slice}_{y}_{z}.tiff"), arr=image_3d, plugin="tifffile")
