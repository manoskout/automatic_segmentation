
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
# import sys
import os
from glob import glob
INPUT_FOLDER = "sample_CGLF/MRI_028_PRO_pCT_CGFL/"

INPUT_FOLDER = "sample_CGLF/MRI_028_PRO_pCT_CGFL/"

# Load the scans in given folder path
def load_scans(path):
    # load the DICOM files
    files = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    files.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    slices = []
    skipcount = 0
    # skip files with no SliceLocation (eg scout views)
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    print("skipped, no SliceLocation: {}".format(skipcount))
    # ensure they are in the correct order
    return sorted(slices, key=lambda s: s.SliceLocation)

def plot_slices(slices):
    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, 187])
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, 187, :])
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[187, :, :].T)
    a3.set_aspect(cor_aspect)
    plt.show()
    
    

##########################################################################
# Multi-slice view code extracted and adapted from: 
# https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(slices, no_axis=False):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    print(img3d.shape)    
    ax1.volume = slices
    ax1.index = len(slices) // 2
    # ax1.imshow(volume[ax1.index],cmap="gray")
    # ax1.set_title(title_volume)
    # ax2.volume = eq_volume
    # ax2.index = len(eq_volume) // 2
    # ax2.imshow(eq_volume[ax2.index],cmap="gray", interpolation="bicubic")
    # ax2.set_title(title_eq_volume)
    
    ax1.imshow(img3d[:, :, ax1.index], cmap="gray")
    ax1.set_aspect(ax_aspect)

    ax2.imshow(img3d[:, ax1.index, :],cmap="gray")
    ax2.set_aspect(sag_aspect)

    ax3.imshow(img3d[ax1.index, :, :].T, cmap="gray")
    ax3.set_aspect(cor_aspect)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


            
def process_key(event):
    # Process key_press events
    fig = event.canvas.figure
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]

    if event.key == 'j':
        previous_slice(ax1)
        previous_slice(ax2)
    elif event.key == 'k':
        next_slice(ax1) 
        next_slice(ax2) 
    fig.canvas.draw()

def previous_slice(ax):
    # Go to the previous slice
    volume = ax.volume
    ax.index = (ax.index-1) % len(volume)
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    # Go to the next slice
    volume = ax.volume
    ax.index = (ax.index+1) % len(volume)
    ax.images[0].set_array(volume[ax.index])
slices = load_scans(INPUT_FOLDER)
multi_slice_viewer(slices)
