from dvc_preprocessing import plot, preprocessing, constants
from skimage.filters import threshold_otsu
import numpy as np


def auto_processing(filename, dirpath='./', data_type=np.int16, init_slice=0, final_slice="last", outname="output", ret="True"):
    '''
    TODO: add outpath
    '''

    stack = preprocessing.read_images_from_h5(filename, data_type, dirpath)

    threshold_value = threshold_otsu(stack)
    print(f'Threshold value: {threshold_value}.')

    stack = preprocessing.intensity_rescaling(stack)

    if data_type == np.int8:
        stack[stack < threshold_value] = constants.INT8MINVAL()
    else:
        stack[stack < threshold_value] = constants.INT16MINVAL()

    if final_slice == "last":
        final_slice = stack.shape[0]

    CoM = preprocessing.volume_CoM(stack, init_slice, final_slice)
    print(f'The center of mass is {CoM}')

    if init_slice != 0 or final_slice != "last":
        stack = preprocessing.crop_around_CoM(
            stack, CoM, (init_slice, final_slice))
    else:
        stack = preprocessing.crop_around_CoM(stack, CoM)

    preprocessing.save_3d_tiff(stack, outname, dirpath)

    if ret == True:
        return stack, CoM, threshold_value
