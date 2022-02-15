#!/usr/bin/env python
# coding: utf-8

from utils import create_summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import skimage.io as skio
from skimage import exposure, filters
from skimage.measure import regionprops
from skimage.feature import corner_fast, corner_peaks
import os
import copy


"""

Routine to preprocess PCT reconstructions to perform DVC analysis

"""

class DVC_Volume:

    def __init__(self, src_file, pixel_size, data_type=np.int16, sample_name="sample", summary=True, summ_dir=None, output_dir=None):

        self.name = sample_name
        self.src_file = src_file
        self.pixel_size = pixel_size
        
        if data_type not in [np.int8, np.int16]:
            raise Warning("Your data might not be suitable for DVC")
            return
        else:
             self.data_type = data_type

        if output_dir == None:
            self.out_dir = src_file.strip(".") + self.name
        else:
            self.out_dir = output_dir

        self.volume = None

        # if summary == True:
        #     create_summary()



    def read_images_from_h5(self):
        """
        TODO: transform it into an auxiliary function
        Returns the stack image of the reconstructed volume
        Ideally you should provide the hdf5 file provided after nabu volume reconstruction for data entry consistency.
        Inputs

        filename - string with the h5 file name 
        data_type - data type from the returned stack image, whether np.int8 or np.int16
        dirpath - the directory where the h5 named 'filename' is found

        """

        with h5py.File(self.src_file, mode = 'r') as h5:
            stack_img = h5['entry0000']['reconstruction']['results']['data'][:]

        self.volume = copy.deepcopy(stack_img)
        return self.volume.astype(self.data_type)


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

#     TODO: check for dimensions; is  it 3d stack or not?
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

    xmin = xcom - (xlen * 0.35)
    xmax = xcom + (xlen * 0.35)
    ymin = ycom - (ylen * 0.25)
    ymax = ycom + (ylen * 0.25)

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



