"""
FastPose: Efficient human pose estimation in depth images with CNN.


Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Angel Martinez Gonzalez <angel.martinez@idiap.ch>

This file is part of the project FastPose for TSCVT paper:

Efficient Convolutional Neural Networks for Depth-Based Multi-Person Pose Estimation

FastPose is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

FastPose is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FastPose. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import scipy.io

import struct



colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],\
            [255, 255, 0], [170, 255, 0], [85, 255, 0],\
            [0, 255, 0], [0, 255, 85], [0, 255, 170],\
            [0, 255, 255], [0, 170, 255], [0, 85, 255],\
            [0, 0, 255], [85, 0, 255], [170, 0, 255],\
            [255, 0, 255], [255, 0, 170], [255, 0, 85]]


##
# @brief Convert mat of floating point to heatmap in uchar8 of three channels.
# @param mat Single chanel floating point matrix.
# @param scaleFactor 
# @return Normalized color heatmap.
#
def float_to_heatmap(mat, scaleFactor=255.0):
    factor=scaleFactor

    heat8U = mat*factor
    heat8U = np.asarray(heat8U, dtype=np.uint8)

    lm = cv2.applyColorMap(heat8U, cv2.COLORMAP_JET)
    return lm



##
# @brief Overlay a heatmap over a given image of the same size.
# @param heatmap Heatmap to be overlayed on the image.
# @param oriImg Original image where to overlay the heatmaps.
# @param alpha Opacity value to weight heatmap in the combination.
# @return Image containing the overlayed heatmap.
#
def get_overlayed_heatmap(heatmap, oriImg, maxval, alpha=0.5):
    [minVal, maxVal, pointMin, pointMax] = cv2.minMaxLoc(heatmap)
    heatMapRes = heatmap 

    scaleFactor = 255.0/maxval

    heat8U =  float_to_heatmap(heatMapRes, scaleFactor)
    oriImg = np.array(oriImg, dtype = np.uint8)
    heat8U = cv2.addWeighted(heat8U, alpha, oriImg, 1.0-alpha, 0)

    return heat8U


##
# @brief Clips depth values greater than the farplane to the 
# value of the farplane depth.
# @param img Depth image to be cliped.
# @param farPlane Farplane depth.
# @return Clipped depth image.
#
def clip_depth_image(img, farPlane, range_list=None):
    s = img.shape
    clipMat = img.copy()

    if len(img.shape) > 2:
        for i in range(3):
            min_, max_= range_list[i][0], range_list[i][1]
            x= clipMat[:,:,i]
            x[x<min_]= min_
            x[x>max_]= max_
            clipMat[:,:,i]= x.copy()

        return clipMat

    clipMat[clipMat < 0.0] = 0.0
    clipMat[clipMat > farPlane] = farPlane

    return clipMat




##
# @brief Convert a floating point depth image to a uchar single channel
# image for visualization purposes. The depth image values are normalized 
# using the farplane's depth.
# @param img Floating point depth image.
# @param farPlane Farplane depth.
# @return Uchar single channel image.
#
def convert_to_uchar(img, farPlane):
    charImg = clip_depth_image(img, farPlane)
    resFactor = 255.0/farPlane
    charImg = charImg*resFactor
    charImg = charImg.astype(np.uint8)

    return charImg




##
# @brief Load depth image given its input extension and data format.
# The method loads the given image into a single channel matrix
# and sets its values in meters. This function has to be used whenever
# try to feed the network with a given input image.
# @param imgPath Image path.
# @return Depth image with values in meters.
#
def load_image(imgPath):
    imgExt = imgPath.split('/')[-1].split('.')[-1]

    try:
        # Load blender image
        if imgExt == 'exr':
            return load_exr_img(imgPath)

        # Load matlab mat file
        elif imgExt == 'mat':
            return load_mat_img(imgPath)

        # Load color image
        elif imgExt == 'jpg' or imgExt == 'png':
            return cv2.imread(imgPath)
            # m= cv2.imread(imgPath, 0)
            m= m.astype(np.float32)/255.*8.
            return m

        # Load numpy file
        elif imgExt == 'npy':
            img = np.load(imgPath)
            img = np.array(img, dtype=np.float32)
            img = img/1000.0 # Transforming into meters
            return img

        elif imgExt == 'tif' or imgExt == 'tiff':
            # print('[INFO] Loading images in tiff format')
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
            img = np.array(img, dtype=np.float32)
            img = img / 1000.0
            return img

        elif imgExt== 'bin':
            return load_binary_file(imgPath)
        else:
            return None
    except IOError:
        print('[ERROR]: Error loading file', imgPath)
        return None
    



##
# @brief Loads exr image that comes from a rendering pipeline.
# @param imgPath Image path.
# @return Single channel depth image with values in meters.
#
def load_exr_img(imgPath):
    # Image in exr format are already in the good range
    img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)

    return img 

##
# Load binary file containing depth image
def load_binary_file(this_file):
    bin_file= open(this_file, "rb")
    ### read rows
    r= bin_file.read(4)
    r= struct.unpack('i',r)[0]
    ### read cols
    c= bin_file.read(4)
    c= struct.unpack('i',c)[0]
    ### read data type
    cv_mat_type= bin_file.read(4)
    cv_mat_type= struct.unpack('i', cv_mat_type)[0]

    img= np.zeros((r,c), dtype=np.float32)

    i=0
    j=0

    while True:
        data= bin_file.read(4)
        if not data:
            break

        data= struct.unpack('f', data)[0]
        img[i,j]= data/1000.

        j+=1
        if j%c == 0:
            j=0
            i+=1

    return img

##
# @brief Loads image stored as a Matlab matrix and sets its values in meters.
# @param imgPath Image path.
# @return Single channel depth image with values in meters.
#
def load_mat_img(imgPath):
    # Images in mat format are normally in milimeters
    try:
        img = scipy.io.loadmat(imgPath)
        img = img['depth']

        if img.dtype!= np.float32:
            img= img.astype(np.float32)

        # img = img / 1000.0
        return img

    except:
        print('[ERROR] Image could not be loaded! Something went wrong.')
        return None



def apply_depth_normalization_single_channel_cube(img):
    farPlane = 8.0
    factor = 1.0/farPlane
    shift  = 0.5
    trans = clip_depth_image(img, farPlane)
    trans =  trans*factor -shift

    #
    trans = trans[np.newaxis, np.newaxis, :, :]

    # print(trans.shape)

    return trans



def apply_depth_normalization_cube(img, farPlane=8.0):
    factor = 1.0/farPlane
    shift  = 0.5

    trans = clip_depth_image(img, farPlane)

    #Transpose image to be on size CxHxW this is the configuration for pytorch
    # And normalize the image to be between [-0.5, 0.5]
    trans = np.repeat(trans[:,:,np.newaxis],3,axis=2)
    trans = trans*factor - shift
    trans = np.transpose(np.float32(trans[:,:,:,np.newaxis]), (3, 2, 0, 1))

    # print(trans.shape) 
    return trans




def apply_rgb_normalization_cube(img):
    #Transpose image to be on size CxHxW this is the configuration for pytorch
    # And normalize the image to be between [-0.5, 0.5]
    # Add the extra channel to fit
    feed = img/255 - 0.5
    feed = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3, 2,0,1))/255 - 0.5

    return feed



def draw_keypoints(img, keypoints, colors=colors):
    for i in range(len(keypoints)):
        p=keypoints[i]
        x=int(round(p[0]))
        y=int(round(p[1]))
        visibility= p[2]

        ### Visibility flag has 3 posible values
        # 0 : annotation is not present
        # 1 : annotation is present and visible
        # 2 : annotation is present but not visible

        if visibility>0.:
            cv2.circle(img, (x,y), 5, colors[i], -1)


#### visualize limb configuration
def draw_limbs(img, keypoints_, limbList_, colors=colors):
    for i in range(len(limbList_)):
        pair= limbList_[i]
        idx1 = pair[0]
        idx2 = pair[1]

        x1 = int(keypoints_[idx1][0])
        y1 = int(keypoints_[idx1][1])
        v1 = int(keypoints_[idx1][2])

        x2 = int(keypoints_[idx2][0])
        y2 = int(keypoints_[idx2][1])
        v2 = int(keypoints_[idx2][2])

        if v2>=1.0 and v1>=1.0:
            cv2.line(img, (x1,y1), (x2,y2), colors[i], 2)














