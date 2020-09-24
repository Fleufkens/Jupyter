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
def get_patches(path, suffix, kernelsize, max_images = None):
      
    patches = [];
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
        patches_tmp, masks_tmp = cut_patches(mask, image, 10, 31)
        patches.extend(patches_tmp)
        masks.extend(masks_tmp)
        
        del(image)
        del(mask)
        
        counter += 1
        
    return [patches, masks]


''' Gets a list of the files in a directory that match the specified suffix       '''
def get_filenames(path, suffix):
    os.chdir(path)
    fns = []
    for fn in os.listdir():
        pos = fn.find('_')
        if fn[pos:] == suffix:
            fns.append(fn)

    return fns
       
                
'''Gets a mask and an image as an input, returns patches which contain brain 
tissue in the middle and are at least 25% tissue .'''
def cut_patches(mask, image, patches_needed = 5, kernelsize = 31):
    #Make list of gt locations in image, if any     
        
    zmax, ymax, xmax = image.shape;
    counter = 0;
    patches = [];
    masks = [];
    attempts = 0;
    while True:
        #Generate random location in the range where the patch will fit in the image
        
        if counter > patches_needed-1:
            break;
        if attempts > 50:
            break;  #Prevent being infinitely stuck
            
        z = np.random.randint(kernelsize, zmax - kernelsize - 1);
        y = np.random.randint(kernelsize, ymax - kernelsize - 1);
        x = np.random.randint(kernelsize, xmax - kernelsize - 1);
        if mask[z,y,x] > 0.4:     #Middle has to be in the mask
            if mask_overlap([z,y,x], mask, kernelsize):     #at least 25% of the patch should contain mask
                patches.append( image[
                        z - kernelsize : z + kernelsize + 1,
                        y - kernelsize : y + kernelsize + 1,
                        x - kernelsize : x + kernelsize + 1]);
                masks.append( mask[
                        z - kernelsize : z + kernelsize + 1,
                        y - kernelsize : y + kernelsize + 1,
                        x - kernelsize : x + kernelsize + 1]);
                counter += 1;
                attempts = 0;
            
            attempts += 1;
    
    return [patches, masks]


'''calculates the overlap of the chosen patch with the brain mask. Returns true if the overlap is > 25%'''
def mask_overlap(loc, mask, kernelsize):
    
    patch = mask[loc[0] - kernelsize : loc[0] + kernelsize,
                 loc[1] - kernelsize : loc[1] + kernelsize,
                 loc[2] - kernelsize : loc[2] + kernelsize];
                 
    return sum(sum(sum(patch>0.5))) > 0.25*(2*kernelsize+1)**3                   


#x,y = get_patches(path, [fn], 31)




        