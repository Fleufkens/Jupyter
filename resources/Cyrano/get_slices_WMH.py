#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:42:20 2018

@author: cyrano
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk


'''loads images from the list of fns, then makes 10 random patches.
    Kernelsize is half the diameter of the kernel, rounded down.'''
def get_slices(path, suffix, max_images = None):
      
    slices = [];
    masks = [];
    
    fns = get_filenames(path, suffix)
    
    counter = 0 
    
    for fn in fns:
        
        if max_images:
            if counter >= max_images:
                break
        
        image = sitk.GetArrayFromImage(
             sitk.ReadImage(os.path.join(path, fn))
             ) 
        
        image = np.swapaxes(image, 0,2)

#        Get mask. Note: mask is not thresholded yet.
        pos = fn.find('_')
        maskfn = fn[:pos]+'_p1T1.nii.gz'
        
        mask = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(path, maskfn)))
        mask = np.swapaxes(mask, 0,2)
                      
        #get 10 patches per image
        slices_tmp, masks_tmp = cut_slices(mask, image, slices_needed = 10, threshold = 0.4)
        slices.extend(slices_tmp)
        masks.extend(masks_tmp)
        
        del(image)
        del(mask)
        
        counter += 1
        
    return [slices, masks]


''' Gets a list of the files in a directory that match the specified suffix       '''
def get_filenames(path, suffix):
    os.chdir(path)
    fns = []
    for fn in os.listdir():
        pos = fn.find('_')
        if fn[pos:] == suffix:
            fns.append(fn)

    return fns
       
                
'''Gets a mask and an image as an input, returns slices which are at least 10% tissue.
    Slices might be sagittal, coronal or transversal.'''
def cut_slices(mask, image, slices_needed = 5, threshold = 0.5):
    #Make list of gt locations in image, if any     
        
    counter = 0;
    max_dim = max(image.shape)
    slices = np.zeros([slices_needed, max_dim, max_dim])
    masks =  np.zeros([slices_needed, max_dim, max_dim])
    attempts = 0;
    while True:
        
        if counter > slices_needed-1:
            break;

        axes = np.random.randint(0,3) #Sagittal, Coronal, Transversal respectively
        idx = np.random.randint(0, image.shape[axes])    
        
        imslice = np.swapaxes(image,0,axes)[idx,:,:]
        maskslice = np.swapaxes(mask,0,axes)[idx,:,:]
            
        if sum(sum(maskslice > threshold)) >= 0.1*np.prod(maskslice.shape): #>10% of slice is filled with tissue
            x_offset = int((max_dim - maskslice.shape[0])/2)
            y_offset = int((max_dim - maskslice.shape[1])/2)
            slices[counter, x_offset:max_dim - x_offset, y_offset : max_dim - y_offset] = imslice
            masks[counter, x_offset:max_dim - x_offset, y_offset : max_dim - y_offset] = maskslice
            counter += 1
            attempts = 0
        
        else:            
            attempts += 1
            if attempts > 50:
                break;  #Prevent being infinitely stuck
    
    return [slices, masks]





        