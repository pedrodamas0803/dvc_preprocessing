import time
import numpy as np
from dvc_preprocessing import preprocessing, plot

t1 = time.time()
path = '/data/id11/3dxrd/ihma206/id11/5min_ys_high_res_/5min_ys_high_res__pct_457MPa_unloading/analysis/'

filename = '5min_ys_high_res__pct_457MPa_unloading_rec.hdf5'

stack = preprocessing.read_images_from_h5(
    filename, data_type=np.int16, dirpath=path)
t2 =time.time()
print(f"{t2-t1}")

print("The middle slice of your stack will appear. Please close it to keep the program running")
#plot.plot_slice_from_stack(stack)
t3=time.time()
print(f"{t3-t2}")
print("The middle slice of your rescaled stack will appear. Please close it to keep the program running")

stack_resc = preprocessing.intensity_rescaling(stack)
#plot.plot_slice_from_stack(stack_resc)
t4=time.time()
print(f"Rescaling done in {t4-t3}!")


print('Calculating the center of mass of the volume')
center = preprocessing.volume_CoM(stack_resc, init_slice=400)
t5=time.time()
print(f"The center of mass (XY plane) is {center}, and it took {t5-t4}.")


print("Cropping your stack around the center of mass (averaged over Z)")
stack_crop = preprocessing.crop_around_CoM(
    stack_resc, CoM=center, xprop=0.25, yprop=0.3)
print("Your image is cropped! Check if you need new xprop/yprop")

print("Calculating the rotation angle of your image.")
angle = preprocessing.get_rotation_angle(stack_resc)
print(f"The mean angle of rotation is: {angle}")

print("Now you can go and try to reconstruct the image accounting for the rotation!")

t2=time.time()
print(f"Your code took {t2-t1} to run.")
