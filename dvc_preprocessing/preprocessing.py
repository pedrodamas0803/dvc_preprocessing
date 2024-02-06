#!/usr/bin/env python
# coding: utf-8

import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import h5py


from scipy.signal import find_peaks
import skimage.io as skio
from skimage import exposure, filters
from skimage.measure import regionprops
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from numpy import linalg as LA

from . import plot as prplot

"""

Routine to preprocess PCT reconstructions to perform DVC analysis

"""


def read_images_from_h5(filename: str, entry: str, out_type=None):
    """
    Returns the stack image of the reconstructed volume
    Ideally you should provide the hdf5 file provided after nabu volume reconstruction for data entry consistency.
    Inputs

    filename - string with the h5 file path
    out_type - data type from the returned stack image

    """

    with h5py.File(filename, "r") as hin:
        if out_type is not None:
            stack_img = hin[entry][:].astype(out_type)
        else:
            stack_img = hin[entry][:]
    return stack_img


def calculate_threshold(image, nbins=256):
    thrs = filters.threshold_otsu(image, nbins)

    return thrs


def crop_slice(image, slice_number, xmin, xmax, ymin, ymax):
    """
    Returns the cropped slices according to the chosen parameters

    Inputs

    image - stack image from which the slice will be taken
    slice_number - well, you know...
    xmin, xmax, ymin, ymax - maximum and minimum values for each axis
    """

    return image[slice_number][ymin:ymax, xmin:xmax]


def crop_stack(image, xmin, xmax, ymin, ymax):
    """
    Returns the cropped stack image, i.e.: crops all slices according to the chosen coordinates

    Inputs

    image - stack image from which the slice will be taken
    slice_number - well, you know...
    xmin, xmax, ymin, ymax - maximum and minimum values for each axis
    """

    return image[:, ymin:ymax, xmin:xmax]


def get_hist(image):
    """
    Get's the histogram of an image.

    Inputs
    image - the image to be used
    plot - if True, plots the histogram

    Returns
    counts - array of counting values for each bin of the histogram
    bins - array of the center of the bins of the histogram
    """
    counts, bins = exposure.histogram(image)

    return counts, bins


def find_peak_position(image, height=1e4, retrn_counts=False):
    """
    Finds the position of the peaks in a histogram of an image.

    Inputs
    image - the image to be used
    height - the height of what should be considered as peaks in the curve
    retrn_counts - if True, returns the frequency associated to that peak position

    Outputs
    peaks - list of the peak positions. If retrn_counts == True, return tuples containing the counts.
    """

    counts, bins = get_hist(image, plot=False)

    peaks, _ = find_peaks(counts, height=height)
    peak_pos = []
    for peak in peaks:
        peak_pos.append(bins[peak])
    if not retrn_counts:
        return peak_pos
    else:
        return [(bins[peak], counts[peak]) for peak in peaks]


def calc_color_lims(img, mult=3):
    """
    Calculates the upper and lower limits to plot an image with centered value on the brightest peak of the histogram.

    Inputs
    image - the image!
    mult - stretching/shrinking factor that defines the amplitude of vmax-vmin by using peak +/- mult*std

    Outputs
    vmin, vmax - tuple with the lower and upper limits.
    """
    peaks = find_peak_position(img)
    vmin = peaks[-1] - mult * img.std()
    vmax = peaks[-1] + mult * img.std()

    return vmin, vmax


def intensity_rescaling(
    image, in_range: str or tuple = "image", mult=3, out_type="uint8"
):
    """
        s the intensity of the stack image according to the chosen percentiles

    Inputs

    image - stack image to be rescaled
    low_perc, high_perc - low and high percentiles to use as parameters to rescale the images
    """

    if in_range != "image":
        imin, imax = calc_color_lims(image, mult=mult)
        rescaled_image = exposure.rescale_intensity(
            image, in_range=(imin, imax), out_range=out_type
        )
    else:
        rescaled_image = exposure.rescale_intensity(
            image, in_range=in_range, out_range=out_type
        )

    return rescaled_image


def _find_center_of_mass(image):
    """
    Finds the center of mass of the image after applying Otsu's thresholding algorithm

    Input
    image - usually a slice of a reconstructed volume

    Output
    coordinates of the weighted center of mass in the form Y, X for consistency with matplotlib.pyplot

    """
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    weighted_center_of_mass = properties[0].weighted_centroid

    return weighted_center_of_mass


def volume_CoM(image, init_slice=0, final_slice="last"):
    """
    Calculates the average coordinates of the center of mass calculated over the range init_slice to final_slice

    Inputs

    image - the stack image that will be iterated over
    init_slice - starting slice for computation of the CoM. default = 0
    final_slice - final slice for computation of CoM. default = len(image)

    Output
    coordinates of the weighted center of mass in the form Y, X for consistency with matplotlib.pyplot

    """
    if final_slice == "last":
        final_slice = len(image[0])

    size = final_slice - init_slice
    x = np.array(np.empty(size))
    y = np.array(np.empty(size))

    for i, img in enumerate(range(init_slice, final_slice)):
        center = _find_center_of_mass(image[img])
        x[i] = center[1]
        y[i] = center[0]

    return np.mean(y), np.mean(x)


def crop_around_CoM(image, CoM: tuple, slices=None, xprop=0.25, yprop=0.25):
    """
    This function will return the image of an slice cut with parameters relative to the calculated center of mass

    Inputs
    image - the image!
    CoM - coordinates of the center of mass as a tuple
    slices - tuple containing the starting
    """
    zlen, xlen, ylen = image.shape

    if slices is None:
        start = int(0)
        end = int(zlen - 1)
    else:
        assert type(slices) == tuple
        start = int(slices[0])
        end = int(slices[1] - 1)

    xcom = CoM[1]
    ycom = CoM[0]

    xmin = int(np.ceil(xcom - (xlen * xprop) // 2))
    xmax = int(np.floor(xcom + (xlen * xprop) // 2))
    ymin = int(np.ceil(ycom - (ylen * yprop) // 2))
    ymax = int(np.floor(ycom + (ylen * yprop) // 2))

    if xmin < 0:
        xmin = int(0)
    elif xmax > xlen:
        xmax = int(xlen - 1)
    else:
        xmin = int(xmin)
        xmax = int(xmax)

    if ymin < 0:
        ymin = int(0)
    elif ymax > ylen:
        ymax = int(ylen - 1)
    else:
        ymin = int(ymin)
        ymax = int(ymax)

    return image[start:end, ymin:ymax, xmin:xmax]


def get_rotation_angle(
    image,
    plot=False,
    canny_sigma=30,
    hough_thrs=5,
    line_len=150,
    line_gap=10,
    mean=True,
):
    """
    This function gets the rotation angle from the image averaged along the Z axis (axis=0 in np).
    Inputs

    image - the stack image
    plot - False by default. If True plots the image together with the detected edges and the calculated lines.
    canny_sigma - the standard deviation for the Gaussian filter used in the Canny edge detector.
    hough_thrs - threshold for the determination of the lines by probabilistic line Hough transform.
    line_len - minimum length accepted for the detected lines.
    line_gap - maximum gap between two pixels to still be considered a line.
    mean - True by default. If set to False it will return the array of calculated angles from the image. True will return the mean of the values.
    """

    # Line finding using the Probabilistic Hough Transform
    img = copy.deepcopy(image)
    img = img.mean(axis=0)
    edges = canny(img, canny_sigma)
    x, y = edges.shape
    edges = edges[int(0.05 * x) : int(0.95 * x), int(0.05 * y) : int(0.95 * y)]
    img = img[int(0.05 * x) : int(0.95 * x), int(0.05 * y) : int(0.95 * y)]
    lines = probabilistic_hough_line(edges, hough_thrs, line_len, line_gap, seed=2)
    prob_angles = []

    for line in lines:
        y = line[1][1] - line[0][1]
        x = line[1][0] - line[0][0]
        vec = x, y
        vec_norm = LA.norm(vec)
        # print(vec, vec_norm)

        if vec[0] != 0 and vec[1] != 0:
            ang = np.rad2deg(np.arctan(y / x))
            prob_angles.append(ang)

    corr_ang = []
    for angle in prob_angles:
        if angle < 0:
            corr_ang.append(angle + 90)
        else:
            corr_ang.append(angle)

    if plot == True:
        prplot.plot_angle_detection(img, edges, lines)

    if not mean:
        return corr_ang
    else:
        return np.mean(np.mean(corr_ang))


def save_3d_tiff(image, filename="output", path="./"):
    """
    Saves a 3D tiff image suitable to perform DVC analysis

    Inputs

    image - stack image to be saved
    filename - 'output' by default, which is a pretty useless name
    path - './' by default
    """
    x, y, z = image.shape
    image_3d = skio.concatenate_images(image)
    skio.imsave(
        os.path.join(path, f"{filename}_{x}_{y}_{z}.tiff"),
        arr=image_3d,
        plugin="tifffile",
    )


def save_3d_subset_tiff(image, init_slice, end_slice, filename, path="./"):
    """
    Saves a subset of slices in a tomo result.

    Inputs

    image - stack image to be sliced and saved
    init_slice - first slice of the stack to be saved as tiff
    end_slice - last slice of the stack to be saved as tiff
    filename - stem for the filename that will be added with the shape of the image
    path - path to the directory where you want to save tour image
    """

    x, y, z = image.shape
    image_3d = skio.concatenate_images(image[init_slice:end_slice])
    skio.imsave(
        os.path.join(path, f"{filename}_{x-init_slice}_{y}_{z}.tiff"),
        arr=image_3d,
        plugin="tifffile",
    )
