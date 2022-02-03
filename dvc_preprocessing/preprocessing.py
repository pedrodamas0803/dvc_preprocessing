#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import skimage.io as skio
from skimage import exposure, filters
from skimage.measure import regionprops
import os

FIGSIZE = (8, 4.5)
INT8MINVAL = int(-128)
INT16MINVAL = int(-32768)

"""

Routine to preprocess PCT reconstructions to perform DVC analysis

"""


# get the image from its path
def read_images_from_h5(filename, data_type=np.int16, dirpath="./"):
    """
    Returns the stack image of the reconstructed volume
    Ideally you should provide the hdf5 file provided after nabu volume reconstruction for data entry consistency.
    Inputs

    filename - string with the h5 file name 
    data_type - data type from the returned stack image, whether np.int8 or np.int16
    dirpath - the directory where the h5 named 'filename' is found

    """

    if data_type not in [np.int8, np.int16]:
        raise Warning("Your data might not be suitable for DVC")
        return

    with h5py.File(os.path.join(dirpath, filename)) as h5:
        stack_img = h5['entry0000']['reconstruction']['results']['data'][:]

    return stack_img.astype(data_type)


# plot selected slice

def plot_slice_from_stack(image, slice_number=None, colormap="gray", figsize=FIGSIZE):
    '''
    Function to plot the selected slice from the stack image. 

    Inputs

    image - the stack image from which the slice will be drawn
    slice_number - well, you can guess what this is. By default (slice_number = None) it will choose the middle slice
    colormap - a color map to be used in the plot. It has to be consistent with the possible values within matplotlib
    '''

    if slice_number == None:
        if len(image[:]) > 1:
            slice_number = int(len(image[:])/2)

    plt.figure(figsize=figsize)
    plt.imshow(image[slice_number], cmap=colormap)
    plt.title(f"Plot of slice {slice_number}")
    plt.show()


def plot_image(image, colormap="gray", name="sample", figsize=FIGSIZE):

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=colormap)
    plt.title(name)
    plt.show()

# plot stack histogram


def plot_histogram(image, hist=False):
    '''
    Plots the intensity histogram for the stack image

    Input

    image - stack image 
    '''

    counts, bins = exposure.histogram(image)
    plt.figure(figsize=FIGSIZE)
    plt.plot(bins, counts, color="red")
    plt.title("Histogram of stack image")
    plt.show()

    if hist == True:
        return counts, bins


# crop the images

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


def find_center_of_mass(image):
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

        center = find_center_of_mass(image[img])
        x[i] = center[1]
        y[i] = center[0]

    return np.mean(y), np.mean(x)


def crop_around_CoM(image, CoM: tuple, slices='all'):
    '''
    This function will return the image of an slice cut with parameters relative to the calculated center of mass

    Inputs
    image - the image!
    CoM - coordinates of the center of mass as a tuple
    slices - tuple containing the starting 
    '''

    zlen, xlen, ylen = image.shape

    if slices != 'all':
        assert type(slices) == tuple
        start = slices[0]
        end = slices[1] + 1
    else:
        start = 0
        end = zlen

    xcom = CoM[1]
    ycom = CoM[0]

    xmin = xcom - (xlen * 0.3)
    xmax = xcom + (xlen * 0.3)
    ymin = ycom - (ylen * 0.4)
    ymax = ycom + (ylen * 0.4)

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


def plot_CoM(image, CoM: tuple):
    '''
    Plots the original slice displaying the coordinates of the center of mass.

    Inputs
    image - again, the slice! 
    CoM - a tuple containing the center of mass (ideally calculated by volume_CoM() for the volume or find_center_of_mass() for a single slice)

    Outputs
    Display the image of a given slice and the coordinates of CoM
    '''

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(CoM[1], CoM[0], s=160, c='C0', marker='+')
    plt.show()


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
    '''

    x, y, z = image.shape
    image_3d = skio.concatenate_images(image[init_slice:end_slice])
    skio.imsave(os.path.join(
        path, f"{filename}_{x}_{y}_{z}.tiff"), arr=image_3d, plugin="tifffile")


def auto_processing(filename, dirpath='./', data_type=np.int16, init_slice=0, final_slice="last", outname="output", ret="True"):
    '''
    TODO: add outpath
    '''

    stack = read_images_from_h5(filename, data_type, dirpath)

    threshold_value = filters.threshold_otsu(stack)
    print(f'Threshold value: {threshold_value}.')

    stack = intensity_rescaling(stack)

    if data_type == np.int8:
        stack[stack < threshold_value] = INT8MINVAL
    else:
        stack[stack < threshold_value] = INT16MINVAL

    if final_slice == "last":
        final_slice = stack.shape[0]

    CoM = volume_CoM(stack, init_slice, final_slice)
    print(f'The center of mass is {CoM}')

    if init_slice != 0 or final_slice != "last":
        stack = crop_around_CoM(stack, CoM, (init_slice, final_slice))
    else:
        stack = crop_around_CoM(stack, CoM)

    save_3d_tiff(stack, outname, dirpath)

    if ret == True:
        return stack, CoM, threshold_value
