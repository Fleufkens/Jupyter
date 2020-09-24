# Author: Cyrano Chatziantoniou
# e-mail: crnchtzntn@gmail.com


import numpy as np
import SimpleITK as sitk
from scipy.interpolate import RegularGridInterpolator 
from multiprocessing import Pool 
import os
from keras.utils import np_utils

'''
TODO: do multiprocessing
'''


''' Example of a generator made with these functions

Receives batch of x and y images, creates a generator that yields augmented patches with random degrees of translation,
rotation, mirroring, noise and contrast modulation.
Y patches should be given unthresholded if possible. 
new_size should contain channels. So an RGB 3D image would have dimensions [x,y,z,3], and a 2d image with 11 channels would be [x,y,11]
if the image has only 1 channel, it should still contain an extra dimension for it

'''

def example_generator(x,y, new_size, batch_size, augmentations_per_image, augment_y = False):
    batch = []
    
    for i in range(batch_size):
        idx = np.random.randint(0, len(x))
        obj = augmentation_obj(x[idx], y[idx], new_size, iterations=augmentations_per_image, augment_y = augment_y)
        batch.append(obj)
        
    #perform augmentations
    augmented_batch = map(perform_augmentations, batch)
    x_aug = np.asarray([i[0] for i in augmented_batch])
    y_aug = np.asarray([i[1] for i in augmented_batch])
    
    perm = np.random.permutation(len(x_aug)) #shuffle x and y
    yield x_aug[perm], y_aug[perm]    
    

    
'''
Object containing images and labels to be augmented. Also holds information on augmentation process.
Used to map the augmentations onto the list of images via multiprocessing.Pool
'''
class augmentation_obj:
    def __init__(self, x, y, new_size, iterations = 10, max_trans = None, max_rot = None, augment_y = True):
        self.x = x                              #Images
        self.y = y                              #Labels
        self.dimensions = x.shape               #dimensions of image
        self.new_size = new_size                #dimensions after augmentation
        self.iterations = iterations            #augmentations per patch
        self.max_translations = max_trans       #translation limit
        self.max_rotations = max_rot            #rotation limit
        self.augment_y = augment_y              #Augment y's
        
    
''' start of the augmentation pipeline. This function can be called with augmentation objects as detaield above.
Suitable for use with generators as well as on batch'''
    
def perform_augmentations(obj):
    
    #Check if augmentation requested
    if obj.iterations <= 0:
        return cut_patch_no_aug(obj)
    
    grid = make_grid(obj.new_size);
    
    #containers for storing augmented images and labels
    X_aug = [None]*obj.iterations
    Y_aug= [None]*obj.iterations
    
    for i in range(obj.iterations):
        #Make random transformation matrix to transform the grid
        transform_matrix = get_transform_matrix(obj.dimensions, obj.new_size, obj.max_translations, obj.max_rotations)
        transform_grid = transform(grid, transform_matrix);
        
        #Resample the image to get the augmented patch
        X_aug[i]= np.squeeze(interpolate(transform_grid, obj.x)) 
        if obj.augment_y:
            Y_aug.append(np.squeeze(interpolate(transform_grid, obj.y)))
        
    return [X_aug, Y_aug]

def cut_patch_no_aug(obj):
    
    xmin = int(obj.dimensions[0]/2) - int(obj.new_size[0]/2)
    ymin = int(obj.dimensions[1]/2) - int(obj.new_size[1]/2)
    xmax = int(obj.dimensions[0]/2) + int(obj.new_size[0]/2)
    ymax = int(obj.dimensions[1]/2) + int(obj.new_size[1]/2)    
    
    #3D
    if len(obj.dimensions) == 3:
        zmin = int(obj.dimensions[2]/2) - int(obj.new_size[2]/2)
        zmax = int(obj.dimensions[2]/2) + int(obj.new_size[2]/2)
        
        x_aug = [obj.x[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]]
        
        if obj.augment_y:
            y_aug = [obj.y[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]]
            return [x_aug, y_aug]
        
        return [x_aug, obj.y]

    #2D
    x_aug = [obj.x[xmin:xmax + 1, ymin:ymax + 1]]
    if obj.augment_y:
        y_aug = [obj.y[xmin:xmax + 1, ymin:ymax + 1]]
        return [x_aug, y_aug]
        
    return [x_aug, obj.y]    

''' Makes a grid based on the input dimensions. Used to take data from the larger patches to make smaller patches. 
    Supports 2D and 3D grids.
    returns a list of the coordinates, with (0,0,0) being the middle'''
def make_grid(dim):
    
    x = np.linspace(-int(dim[0]/2), int(dim[0]/2), dim[0])
    y = np.linspace(-int(dim[1]/2), int(dim[1]/2), dim[1])


    if len(dim) == 4:
        z = np.linspace(-int(dim[2]/2), int(dim[2]/2), dim[2])
        xv, yv, zv = np.meshgrid(x,y,z, indexing='ij')
        grid = np.transpose((xv, yv, zv, np.ones(xv.shape)))
        
        gridarray = [grid for _ in range(dim[-1])]
        grid = np.swapaxes(np.stack(gridarray,axis=4),3,4)
        grid = np.swapaxes(grid, 0,2) #if you find a way to do this without the swapaxes (in the same time), let me know!
    
    else: 
        if len(dim) == 3:
        
            xv, yv = np.meshgrid(x,y, indexing='ij')
            grid = np.transpose((xv, yv, np.ones(xv.shape)))
            
            gridarray = [grid for _ in range(dim[-1])]
            grid = np.swapaxes(np.stack(gridarray,axis=3),2,3)
            grid = np.swapaxes(grid, 0,1)
            
        else: raise Exception('New patch shape dimensions should be 2D or 3D')

    return grid
    
'''Transforms the grid according to the transformation matrix'''
def transform(grid, transform_matrix):
    
    shape = list(grid.shape)
    new_shape = np.copy(shape)
    new_shape[-1] -= 1
    
    grid_reshaped = np.reshape(grid, [-1, grid.shape[-1]])  
    grid_transformed = np.dot(transform_matrix, np.transpose(grid_reshaped))
    grid_transposed = np.transpose(grid_transformed[:new_shape[-1],:])
    grid_transposed_reshaped = np.reshape(grid_transposed,new_shape)
        
    return grid_transposed_reshaped;

        
'''Returns an random transformation matrix which transforms the grid such that it fits into the old image'''
#TODO: add limits
#TODO: Add checks for rotation
def get_transform_matrix(old_size, new_size, max_trans, max_rot):
    if len(new_size) == 4:
        transform_matrix = np.eye(4)
        transform_matrix = translate3D(
                                rotate3D(
                                        mirror3D(transform_matrix),
                                        old_size, new_size, max_rot),
                                old_size, new_size, max_trans)

    else: 
        transform_matrix = np.eye(3)        
        transform_matrix = translate2D(
                                rotate2D(
                                        mirror2D(transform_matrix),
                                        old_size, new_size, max_rot),
                                old_size, new_size, max_trans);
        
    return transform_matrix
    

def interpolate(grid, patch):
    
    x = np.linspace( -int(patch.shape[0]/2), int(patch.shape[0]/2), patch.shape[0])
    y = np.linspace( -int(patch.shape[1]/2), int(patch.shape[1]/2), patch.shape[1])
    if len(grid[0,0]) > 1:
        z = np.linspace( -int(patch.shape[2]/2), int(patch.shape[2]/2), patch.shape[2])
        interpolating_func = RegularGridInterpolator((x,y,z), patch)
    else:
        interpolating_func = RegularGridInterpolator((x,y,), patch)
    
    interpolated_patch = np.squeeze(interpolating_func(grid))
    
    return interpolated_patch;

def gamma_mod(patch):
    gamma = np.random.uniform(low=0.8, high=1.2)
    patch = patch ** (1/gamma)
    return patch

def contrast_mod(patch):
#    contrast_factor = np.random.uniform(low=0.8, high=1.2)
#    patch = patch * 255.
#
#    image = Image.fromarray(patch.astype('uint8'), 'RGB')
#    contrast = ImageEnhance.Contrast(image)
#    contrast = contrast.enhance(contrast_factor)
#
#    patch = np.array(contrast)
#    patch = patch / 255.

    return patch


    
'''Adds a translation to the transformation matrix along the z, y, and x directions. '''
def translate3D(transform_matrix, old_patch_size, new_patch_size, max_translations):
    #Find max distance possible / allowed by user
    maxlength = 0.5*np.sqrt(new_patch_size[0]**2 + new_patch_size[1]**2 + new_patch_size[2]**2)
    maxdx = int(old_patch_size[0]/2 - maxlength)
    maxdy = int(old_patch_size[1]/2 - maxlength)
    maxdz = int(old_patch_size[2]/2 - maxlength)
    
    if max_translations != None:
        maxdx = min(maxdx, max_translations[0])
        maxdy = min(maxdy, max_translations[1])
        maxdz = min(maxdz, max_translations[2])
        
    #Apply to transformation matrix            
    translate_matrix = np.eye(4);
    if maxdx > 0:
        translate_matrix[0,3] = np.random.randint(-maxdx, maxdx)
    else:
        translate_matrix[0,3] = 0
        
    if maxdy > 0:
        translate_matrix[1,3] = np.random.randint(-maxdy, maxdy)
    else:
        translate_matrix[1,3] = 0
        
    if maxdz > 0:
        translate_matrix[2,3] = np.random.randint(-maxdz, maxdz)
    else:
        translate_matrix[2,3] = 0
      
    return np.float32(np.dot(translate_matrix, transform_matrix))



def translate2D(transform_matrix, old_sz, new_sz, max_translations):
    #Find max distance possible / allowed by user
    maxlength = 0.5*np.sqrt(new_sz[0]**2 + new_sz[1]**2)
    maxdx = int(old_sz[0]/2 - maxlength)
    maxdy = int(old_sz[1]/2 - maxlength)
        
    if max_translations != None:
        maxdx = min(maxdx, max_translations[0])
        maxdy = min(maxdy, max_translations[1])
        
        if sum(max_translations) == 0:
            return transform_matrix
    
    translate_matrix = np.eye(3)
    if maxdx > 0:
        translate_matrix[0, 2] = np.random.randint(-maxdx, maxdx)
    if maxdy > 0:
        translate_matrix[1, 2] = np.random.randint(-maxdy, maxdy)

    return np.float32(np.dot(translate_matrix, transform_matrix))

'''Adds a rotation to the transformation matrix according to random euler_angles
   Naive method: Calculates a random angle and then sees if it's allowed. Doesn't take too long if there aren't many restrictions.'''
   
def rotate3D(transform_matrix, old_sz, new_sz, max_rotations ):
    
    if max_rotations != None:
        if sum(max_rotations) >0:
            return transform_matrix
    
    x = int(new_sz[0]/2)
    y = int(new_sz[1]/2)
    z = int(new_sz[2]/2)
    x2 = int(old_sz[0]/2)
    y2 = int(old_sz[1]/2)
    z2 = int(old_sz[2]/2)
    
    corners = [ [x,y,z,1],
                [x,-y,z,1],
                [-x,y,z,1],
                [-x,-y,z,1] ]
    
    cntr = 0
    while True:
           
        [phi, theta, psi] = euler_angles(max_rotations);
        rot_matrix = make_rot_matrix3D(phi,theta,psi);
        new_tm = np.float32(np.dot(rot_matrix, transform_matrix))
        
        if np.all(abs(np.dot(new_tm, np.transpose(corners))) <= np.transpose(np.tile([x2,y2,z2,1],(4,1)))): #check if any of the corners 
            return new_tm
        cntr += 1
        if cntr > 5000:
            raise Exception("It's taking too long to fit a rotated patch into your image. Try increasing image size, decreasing patch size, or providing some maximum rotation.")
    
def rotate2D(transform_matrix, old_sz, new_sz, max_rot):
    
    a = int(new_sz[0]/2)
    b = int(new_sz[1]/2)
    A = int(old_sz[0]/2)
    B = int(old_sz[1]/2)
    
    #Calculate max theta over which can be rotated
    if A <= np.sqrt(a**2 + b**2):
        maxThetaX = np.arccos( A / np.sqrt(a**2 + b**2)) - np.pi/4
    else: 
        maxThetaX = np.pi
    if B <= np.sqrt(a**2 + b**2):
        maxThetaY = np.arccos( B / np.sqrt(a**2 + b**2)) - np.pi/4
    else: 
        maxThetaY = np.pi
    
    maxTheta = abs(np.nanmin([maxThetaX, maxThetaY])    )
    if np.isnan(maxTheta) :
        maxTheta = np.pi
    
    if max_rot != None:
        maxTheta = min(maxTheta, max_rot)
         
        if max_rot == 0: #no rotations
            return(transform_matrix)        
    
    theta = np.random.uniform(-maxTheta, maxTheta)
    rot_matrix = make_rot_matrix2D(theta);

    return np.float32(np.dot(rot_matrix, transform_matrix))


'''Adds a reflection to the transformation matrix depending on a coin flip per axis'''
def mirror3D(transform_matrix):
    mirror_matrix = np.eye(4);
    for i in range(3):
        if np.random.rand()>0.5:
            mirror_matrix[i,i] = -1;
        
    return np.float32(np.dot(mirror_matrix, transform_matrix))

def mirror2D(transform_matrix):
    mirror_matrix = np.eye(3);
    for i in range(2):
        if np.random.rand() > 0.5:
            mirror_matrix[i, i] = -1

    return np.float32(np.dot(mirror_matrix, transform_matrix))

''' generate uniform theta and phi'''
def get_angles():
#    if max_rot != None:
#        max_angles = max_rot
#    else:
#        max_angles = [2*np.pi]*2
        
    #Generate random vector in 3D to sample uniform locations on the sphere
    x = np.random.uniform(-1,1)
    y = np.random.uniform(-1,1)
    z = np.random.uniform(-1,1)
    
    length = np.sqrt(x**2 + y**2 + z**2)
    x /= length
    y /= length
    z /= length
    
    theta = np.arccos(z)
    phi = np.arctan(y/x)
    
    return(theta,phi)
    
'''    Generate euler angles to be used in the rotation. Returns [Theta, Phi, Psi]'''
def euler_angles(max_rot):

    if max_rot != None:
        max_angles = max_rot
    else:
        max_angles = [2*np.pi]*3
    
    #    Uniform:
    phi = np.random.uniform(0, max_angles[0]);
    theta = np.random.uniform (0, max_angles[1]);
    psi = np.random.uniform(0, max_angles[2]);
    
    return [phi,theta,psi]
    
    
'''Returns a rotation matrix when given 3 euler angles    '''
def make_rot_matrix3D(x,y,z):
    X = np.eye(4);
    Y = np.eye(4);
    Z = np.eye(4);
    
    X[1,1] = np.cos(x);
    X[1,2] = -np.sin(x);
    X[2,1] = np.sin(x);
    X[2,2] = np.cos(x);

    Y[0,0] = np.cos(y);
    Y[0,2] = np.sin(y);
    Y[2,0] = -np.sin(y);
    Y[2,2] = np.cos(y);
    
    Z[0,0] = np.cos(z);
    Z[0,1] = -np.sin(z);
    Z[1,0] = np.sin(z);
    Z[1,1] = np.cos(z);

    return np.dot(X, np.dot(Y,Z)) 

def make_rot_matrix2D(theta):
    rot_matrix = np.eye(3)

    rot_matrix[0, 0] = np.cos(theta)
    rot_matrix[0, 1] = np.sin(theta)
    rot_matrix[1, 0] = -np.sin(theta)
    rot_matrix[1, 1] = np.cos(theta)

    return rot_matrix


    
# =============================================================================
# Checks & Error processing
# =============================================================================


def check_dim(x,y, new_sz):
    if x.shape != y.shape:
        raise Exception("Mismatch of image and label dimensions. x-shape = {xshape}, y-shape = {yshape} ".format(xshape=x.shape, yshape = y.shape))
    
    for i in range(len(new_sz)):
        if x.shape[i] < new_sz[i]:
            raise Exception("new_size dimensions must be smaller or equal to dimensions of image. Error in dimension {dim}, image dimension = {imdim}, new_sisze dimension = {new_dim}".format(
                    dim = i, imdim=x.shape[i], newdim = new_sz[i]))
            break;
            
                













