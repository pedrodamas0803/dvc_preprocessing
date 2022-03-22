#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import h5py
import skimage.io as skio
from skimage import exposure, filters
from skimage.util import img_as_int, img_as_ubyte
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
    if data_type not in ['int8', 'int16']:
        raise TypeError("Your image won't be suitable for DVC analysis. Choose between data_type = int8 or int16.")

    with h5py.File(os.path.join(dirpath, filename)) as h5:
        stack_img = h5['entry0000']['reconstruction']['results']['data'][:]

    if data_type == 'int8':

        return img_as_ubyte(stack_img)
        
    elif data_type == 'original':
        
        return stack_img
        
    else:

        return img_as_int(stack_img)

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


def _find_center_of_mass(image):
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


def volume_CoM(image, init_slice=0, final_slice='last'):
    '''
    Calculates the average coordinates of the center of mass calculated over the range init_slice to final_slice

    Inputs

    image - the stack image that will be iterated over
    init_slice - starting slice for computation of the CoM. default = 0
    final_slice - final slice for computation of CoM. default = len(image)

    Output
    coordinates of the weighted center of mass in the form Y, X for consistency with matplotlib.pyplot

    '''
    if final_slice == 'last':
        final_slice = len(image[0])

    size = final_slice - init_slice
    x = np.array(np.empty(size))
    y = np.array(np.empty(size))

    for i, img in enumerate(range(init_slice, final_slice)):

        center = _find_center_of_mass(image[img])
        x[i] = center[1]
        y[i] = center[0]

    return np.mean(y), np.mean(x)


def crop_around_CoM(image, CoM: tuple, slices='all', xprop=0.25, yprop=0.25):
    '''
    This function will return the image of an slice cut with parameters relative to the calculated center of mass

    Inputs
    image - the image!
    CoM - coordinates of the center of mass as a tuple
    slices - tuple containing the starting 
    '''
    zlen, xlen, ylen = image.shape
    #if image.shape == 3:
    #   zlen, xlen, ylen = image.shape
    #else:
    #    xlen, ylen = image.shape
    #    zlen = 0

    if slices != 'all':
        assert type(slices) == tuple
        start = slices[0]
        end = slices[1] + 1
    else:
        start = 0
        end = zlen

    xcom = CoM[1]
    ycom = CoM[0]

    xmin = xcom - (xlen * xprop)
    xmax = xcom + (xlen * xprop)
    ymin = ycom - (ylen * yprop)
    ymax = ycom + (ylen * yprop)

    if xmin < 0:
        xmin = 0
    elif xmax > xlen:
        xmax = xlen
    else:
        xmin = int(xmin)
        xmax = int(xmax)

    if ymin < 0:
        ymin = 0
    elif ymax > ylen:
        ymax = ylen
    else:
        ymin = int(ymin)
        ymax = int(ymax)

    return image[start:end, ymin:ymax, xmin:xmax]


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
