#!/usr/bin/env python
"""
Utility functions for creating qc images for use in html pages. Adapted from
functions in qc-html.py from the datman repo (https://github.com/TIGRLab/datman)
"""

import os
import logging
from copy import copy
import nibabel as nib
import numpy as np
import scipy as sp
import matplotlib
import scipy.signal as sig
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt

FIGDPI = 144
LOGGER = logging.getLogger('qc_pages.imgs')

def fmri_plots(func_nii, mask_nii, motion1D_path, plot_title, output_name):
    """
    func_nii        The absolute path to a functional nifti image
    mask_nii        The absolute path to a mask for func_nii
    motion1D_path   The full path to the motion.1D file
    plot_title      The string title to add above the figure
    output_name     The full path and name for the output .png image

    Calculates and plots:
         - Mean and SD of normalized spectra across brain.
         - Framewise displacement (mm/TR) of head motion.
         - Mean correlation from 10% of the in-brain voxels.
    """
    ##############################################################################
    # spectra

    plt.subplot(2,2,1)
    func = load_masked_data(func_nii, mask_nii)
    spec = sig.detrend(func, type='linear')
    spec = sig.periodogram(spec, fs=0.5, return_onesided=True, scaling='density')
    freq = spec[0]
    spec = spec[1]
    sd = np.nanstd(spec, axis=0)
    mean = np.nanmean(spec, axis=0)

    plt.plot(freq, mean, color='black', linewidth=2)
    plt.plot(freq, mean + sd, color='black', linestyle='-.', linewidth=0.5)
    plt.plot(freq, mean - sd, color='black', linestyle='-.', linewidth=0.5)
    plt.title('Whole-brain spectra mean, SD', size=6)
    plt.xticks(size=6)
    plt.yticks(size=6)
    plt.xlabel('Frequency (Hz)', size=6)
    plt.ylabel('Power', size=6)
    plt.xticks([])

    ##############################################################################
    # framewise displacement
    plt.subplot(2,2,2)
    fd_thresh = 0.5
    motion = np.genfromtxt(motion1D_path)
    motion[:,0] = np.radians(motion[:,0]) * 50 # 50 = head radius, need not be constant.
    motion[:,1] = np.radians(motion[:,1]) * 50 # 50 = head radius, need not be constant.
    motion[:,2] = np.radians(motion[:,2]) * 50 # 50 = head radius, need not be constant.
    motion = np.abs(np.diff(motion, n=1, axis=0))
    motion = np.sum(motion, axis=1)
    t = np.arange(len(motion))

    plt.plot(t, motion.T, lw=1, color='black')
    plt.axhline(y=fd_thresh, xmin=0, xmax=len(motion), color='r')
    plt.xlim((-3, len(motion) + 3)) # this is in TRs
    plt.ylim(0, 2) # this is in mm/TRs
    plt.xticks(size=6)
    plt.yticks(size=6)
    plt.xlabel('TR', size=6)
    plt.ylabel('Framewise displacement (mm/TR)', size=6)
    plt.title('Head motion', size=6)

    ##############################################################################
    # whole brain correlation
    plt.subplot(2,2,3)
    idx = np.random.choice(func.shape[0], func.shape[0]/10, replace=False)
    corr = func[idx, :]
    corr = sp.corrcoef(corr, rowvar=1)
    mean = np.mean(corr, axis=None)
    std = np.std(corr, axis=None)

    im = plt.imshow(corr, cmap=plt.cm.RdBu_r, interpolation='nearest', vmin=-1, vmax=1)
    plt.xlabel('Voxel', size=6)
    plt.ylabel('Voxel', size=6)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar(im)
    cb.set_label('Correlation (r)', labelpad=0, y=0.5, size=6)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontsize(6)
    plt.title('Whole-brain r mean={}, SD={}'.format(str(mean), str(std)), size=6)
    plt.suptitle(plot_title)
    plt.savefig(output_name, format='png', dpi=FIGDPI)
    plt.close()

def load_masked_data(func_nii, mask_nii):
    """
    Accepts 'functional.nii.gz' and 'mask.nii.gz', and returns a voxel's x
    timepoints matrix of the functional data in non-zero mask locations.
    """

    func = nib.load(func_nii).get_data()
    mask = nib.load(mask_nii).get_data()

    mask = mask.reshape(mask.shape[0]*mask.shape[1]*mask.shape[2])
    func = func.reshape(func.shape[0]*func.shape[1]*func.shape[2],
                                                    func.shape[3])

    # find within-brain timeseries
    idx = np.where(mask > 0)[0]
    func = func[idx, :]

    return func

def montage(image, name, nii_name, pic, cmaptype='grey', mode='3d', minval=None, maxval=None, box=None):
    """
    Creates a montage of images displaying an image set on top of a grayscale
    image.

    Generally, this will be used to plot an image (of type 'name') that was
    generated from the original file 'filename'. So if we had an SNR map
    'SNR.nii.gz' from 'fMRI.nii.gz', we would submit everything to montage
    as so:

        montage('SNR.nii.gz', 'SNR', 'EPI.nii.gz', 'EPI_SNR.png')

    Usage:
        montage(image, name, nii_name, pic)

        image     -- submitted image file name
        name      -- name of the printout (e.g, SNR map, t-stats, etc.)
        nii_name  -- qc image file name
        pic       -- Path to save the figure .png to
        cmaptype  -- 'redblue', 'hot', or 'gray'.
        minval    -- colormap minimum value as a % (None == 'auto')
        maxval    -- colormap maximum value as a % (None == 'auto')
        mode      -- '3d' (prints through space) or '4d' (prints through time)
        box       -- a (3,2) tuple that describes the start and end voxel
                     for x, y, and z, respectively. If None, we find it ourselves.
    """

    image = str(image) # input checks
    opath = os.path.dirname(image) # grab the image folder
    output = str(image)
    image = nib.load(image).get_data() # load in the daterbytes

    if mode == '3d':
        if len(image.shape) > 3: # if image is 4D, only keep the first time-point
            image = image[:, :, :, 0]

        image = np.transpose(image, (2,0,1))
        image = np.rot90(image, 2)

        # use bounding box (submitted or found) to crop extra-brain regions
        if box == None:
            box = bounding_box(image) # get the image bounds
        elif box.shape != (3,2): # if we did, ensure it is the right shape
            LOGGER.error('ERROR: Bounding box should have shape = (3,2).')
            raise ValueError

        image = image[box[0,0]:box[0,1], box[1,0]:box[1,1], box[2,0]:box[2,1]]
        steps = np.round(np.linspace(0,np.shape(image)[0]-2, 36)) # coronal plane
        factor = 6

    if mode == '4d':
        image = reorient_4d_image(image)
        midslice = np.floor((image.shape[2]-1)/2) # print a single plane across all slices
        factor = np.ceil(np.sqrt(image.shape[3])) # print all timepoints
        factor = factor.astype(int)

    # colormapping -- set value
    if cmaptype == 'redblue': cmap = plt.cm.RdBu_r
    elif cmaptype == 'hot': cmap = plt.cm.OrRd
    elif cmaptype == 'gray': cmap = plt.cm.gray
    else:
        LOGGER.debug('No valid colormap supplied, default = greyscale.')
        cmap = plt.cm.gray

    # colormapping -- set range
    if minval == None:
        minval = np.min(image)
    else:
        minval = np.min(image) + ((np.max(image) - np.min(image)) * minval)

    if maxval == None:
        maxval = np.max(image)
    else:
        maxval = np.max(image) * maxval

    cmap.set_bad('g', 0)  # value for transparent pixels in the overlay

    fig, axes = plt.subplots(nrows=factor, ncols=factor, facecolor='white')
    for i, ax in enumerate(axes.flat):

        if mode == '3d':
            im = ax.imshow(image[steps[i], :, :], cmap=cmap, interpolation='nearest', vmin=minval, vmax=maxval)
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        elif mode == '4d' and i < image.shape[3]:
            im = ax.imshow(image[:, :, midslice, i], cmap=cmap, interpolation='nearest')
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        elif mode == '4d' and i >= image.shape[3]:
            ax.set_axis_off() # removes extra axes from plot

    plt.subplots_adjust(left=0, right=0.85, top=0.9, bottom=0)

    cbar_ax = fig.add_axes([0.88, 0.10, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(nii_name + '\n' + name, size=10)

    fig.savefig(pic, format='png', dpi=FIGDPI)
    plt.close()

def bounding_box(image_3D_array):
    """
    Finds a box that only includes all nonzero voxels in a 3D image. Output box
    is represented as 3 x 2 numpy array with rows denoting x, y, z, and columns
    denoting stand and end slices.

    Usage:
        box = bounding_box(image_3D_array)
    """

    # find 3D bounding box
    box = np.zeros((3,2))  # init bounding box
    flag = 0  # ascending

    for i, dim in enumerate(image_3D_array.shape): # loop through (x, y, z)

        # ascending search
        while flag == 0:
            for dim_test in np.arange(dim):

                # get sum of all values in each slice
                if i == 0:   test = np.sum(image_3D_array[dim_test, :, :])
                elif i == 1: test = np.sum(image_3D_array[:, dim_test, :])
                elif i == 2: test = np.sum(image_3D_array[:, :, dim_test])

                # if slice is nonzero, set starting bound, switch to descending
                if test >= 1:
                    box[i, 0] = dim_test
                    flag = 1
                    break

        # descending search
        while flag == 1:
            for dim_test in np.arange(dim):

                dim_test = dim-dim_test - 1  # we have to reverse things

                # get sum of all values in each slice
                if i == 0:   test = np.sum(image_3D_array[dim_test, :, :])
                elif i == 1: test = np.sum(image_3D_array[:, dim_test, :])
                elif i == 2: test = np.sum(image_3D_array[:, :, dim_test])

                # if slice is nonzero, set ending bound, switch to ascending
                if test >= 1:
                    box[i, 1] = dim_test
                    flag = 0
                    break

    return box

def reorient_4d_image(image):
    """
    Reorients the data to radiological, one TR at a time
    """
    for i in np.arange(image.shape[3]):

        if i == 0:
            newimage = np.transpose(image[:, :, :, i], (2,0,1))
            newimage = np.rot90(newimage, 2)

        elif i == 1:
            tmpimage = np.transpose(image[:, :, :, i], (2,0,1))
            tmpimage = np.rot90(tmpimage, 2)
            newimage = np.concatenate((newimage[...,np.newaxis],
                                       tmpimage[...,np.newaxis]), axis=3)

        else:
            tmpimage = np.transpose(image[:, :, :, i], (2,0,1))
            tmpimage = np.rot90(tmpimage, 2)
            newimage = np.concatenate((newimage,
                                       tmpimage[...,np.newaxis]), axis=3)

    image = copy(newimage)

    return image

def find_epi_spikes(image, nii_name, pic, bvec=None):
    """
    Plots, for each axial slice, the mean instensity over all TRs.
    Strong deviations are an indication of the presence of spike
    noise.

    If bvec is supplied, we remove all time points that are 0 in the bvec
    vector.

    Usage:
        find_epi_spikes(image, nii_name, pic)

        image     -- submitted image file name
        nii_name  -- qc image file name
        pic       -- path to save the .png figure to
        bvec      -- numpy array of bvecs (for finding direction = 0)

    """

    image = str(image)             # input checks
    opath = os.path.dirname(image) # grab the image folder

    # load in the daterbytes
    output = str(image)
    image = nib.load(image).get_data()
    image = reorient_4d_image(image)

    x = image.shape[1]
    y = image.shape[2]
    z = image.shape[0]
    t = image.shape[3]

    # initialize the spikecount
    spikecount = 0

    # find the most square set of factors for n_trs
    factor = np.ceil(np.sqrt(z))
    factor = factor.astype(int)

    fig, axes = plt.subplots(nrows=factor, ncols=factor, facecolor='white')

    # sets the bounds of the image
    c1 = np.round(x*0.25)
    c2 = np.round(x*0.75)

    # for each axial slice
    for i, ax in enumerate(axes.flat):
        if i < z:

            v_mean = np.array([])
            v_sd = np.array([])

            # find the mean, STD, of each dir and concatenate w. vector
            for j in np.arange(t):

                # gives us a subset of the image
                sample = image[i, c1:c2, c1:c2, j]
                mean = np.mean(sample)
                sd = np.std(sample)

                if j == 0:
                    v_mean = copy(mean)
                    v_sd = copy(sd)
                else:
                    v_mean = np.hstack((v_mean, mean))
                    v_sd = np.hstack((v_sd, sd))

            # crop out b0 images
            if bvec is None:
                v_t = np.arange(t)
            else:
                idx = np.where(bvec != 0)[0]
                v_mean = v_mean[idx]
                v_sd = v_sd[idx]
                v_t = np.arange(len(idx))

            # keep track of spikes
            v_spikes = np.where(v_mean > np.mean(v_mean)+np.mean(v_sd))[0]
            spikecount = spikecount + len(v_spikes)

            ax.plot(v_mean, color='black')
            ax.fill_between(v_t, v_mean-v_sd, v_mean+v_sd, alpha=0.5, color='black')
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        else:
            ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.suptitle('{}\nDTI Slice/TR Wise Abnormalities'.format(nii_name), size=10)


    fig.savefig(pic, format='png', dpi=FIGDPI)
    plt.close()
