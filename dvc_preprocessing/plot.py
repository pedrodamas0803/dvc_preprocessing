#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure
from . import constants as const

def plot_image(image, colormap="gray", name="sample", figsize=const.FIGSIZE()):

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=colormap)
    plt.title(name)
    plt.show()

def plot_slice_from_stack(image, slice_number=None, colormap="gray", figsize=const.FIGSIZE()):
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
    
def plot_histogram(image, hist=False, adjust=True, figsize=const.FIGSIZE()):
    '''
    Plots the intensity histogram for the stack image

    Input

    image - stack image 
    '''

    counts, bins = exposure.histogram(image)
    if adjust:
        counts[0]=counts[1]
        counts[-1] = counts[-2]
        
    plt.figure(figsize=figsize)
    plt.plot(bins, counts, color="red")
    plt.title("Histogram of stack image")
    plt.show()

    if hist == True:
        return counts, bins
    
def plot_CoM(image, CoM: tuple, cmap='gray'):
    '''
    Plots the original slice displaying the coordinates of the center of mass.

    Inputs
    image - again, the slice! 
    CoM - a tuple containing the center of mass (ideally calculated by volume_CoM() for the volume or find_center_of_mass() for a single slice)
    cmap - colormap to use when plotting the image, you can choose from matplotlib.pyplot options

    Outputs
    Display the image of a given slice and the coordinates of CoM
    '''

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(CoM[1], CoM[0], s=160, c='C0', marker='+', cmap='gray')
    plt.show()

def plot_angle_detection(image, edges, lines):
    # Generating figure 2
    
    if len(lines) == 0:
        return 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title('Canny edges and detected lines')

    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    
    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    

