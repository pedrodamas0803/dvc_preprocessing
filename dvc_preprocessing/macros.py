import dvc_preprocessing.preprocessing import * as pp

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