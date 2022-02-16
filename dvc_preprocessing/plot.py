#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as cm
import os
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
    
def plot_histogram(image, hist=False, figsize=const.FIGSIZE()):
    '''
    Plots the intensity histogram for the stack image

    Input

    image - stack image 
    '''

    counts, bins = exposure.histogram(image)
    plt.figure(figsize=figsize)
    plt.plot(bins, counts, color="red")
    plt.title("Histogram of stack image")
    plt.show()

    if hist == True:
        return counts, bins
    
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

def plot_lines(image, edges, lines):
    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    

