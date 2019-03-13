################################################################################

# image manipulation utilities - general and stereo camera specific

# Copyright (c) 2018 Toby Breckon, Durham University, UK

# License: MIT License (MIT)

################################################################################

import cv2
import numpy as np

################################################################################

# concatenate two RGB/grayscale images horizontally (left to right) handling
# differing channel numbers or image heights in the input

def h_concatenate(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0];
    width1 = img1.shape[1];
    if (len(img1.shape) == 2):
        channels1 = 1;
    else:
        channels1 = img1.shape[2];

    height2 = img2.shape[0];
    width2 = img2.shape[1];
    if (len(img2.shape) == 2):
        channels2 = 1;
    else:
        channels2 = img2.shape[2];

    # make all images 3 channel, or assume all same channel

    if ((channels1 > channels2) and (channels1 == 3)):
        out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR);
        out1 = img1;
    elif ((channels2 > channels1) and (channels1 == 3)):
        out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR);
        out2 = img2;
    else: # both must be equal
        out1 = img1;
        out2 = img2;

    # height of first image is master height, width can remain unchanged

    if (height1 != height2):
        out2 = cv2.resize(out2, (height1, width2))

    return np.hstack((out1, out2));

################################################################################

# concatenate two RGB/grayscale images vertically (top to bottom) handling
# differing channel numbers or image heights in the input

def v_concatenate(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0];
    width1 = img1.shape[1];
    if (len(img1.shape) == 2):
        channels1 = 1;
    else:
        channels1 = img1.shape[2];

    height2 = img2.shape[0];
    width2 = img2.shape[1];
    if (len(img2.shape) == 2):
        channels2 = 1;
    else:
        channels2 = img2.shape[2];

    # make all images 3 channel, or assume all same channel

    if ((channels1 > channels2) and (channels1 == 3)):
        out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR);
        out1 = img1;
    elif ((channels2 > channels1) and (channels1 == 3)):
        out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR);
        out2 = img2;
    else: # both must be equal
        out1 = img1;
        out2 = img2;

    # width of first image is master height, height can remain unchanged

    if (width1 != width2):
        out2 = cv2.resize(out2, (height2, width1))

    return np.vstack((out1, out2));

################################################################################

# calculate new camera matrices and rectification maps 

def initCalibration(Kl, Kr, distCoeffsL, distCoeffsR, height, width, R, T):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(Kl, distCoeffsL, Kr, distCoeffsR, (width, height), R, T, cv2.CALIB_ZERO_DISPARITY, 0, (width, height))

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(Kl, distCoeffsL, R1, P1, (width, height), cv2.CV_32FC2)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(Kr, distCoeffsL, R2, P2, (width, height), cv2.CV_32FC2)

    return P1, P2, map_l_x, map_l_y, map_r_x, map_r_y