import os
import argparse
import requests
import configparser
import time

import cv2
import open3d
import math
import numpy as np

from collections import OrderedDict
from camera_stream import *
from zed_calibration import *
from utils import *

class Odometry:
    def __init__(self, Q, height, width, fx, cx, cy):
        self.height = height
        self.width = width
        self.fx = fx
        self.cx = cx
        self.cy = cy
        
        self.prevLeft = None
        self.prevRight = None
        self.prevDisparity = None
        self.prevOdometry = np.identity(4)

        self.curLeft = None
        self.curRight = None
        self.curDisparity = None

        self.Q = Q
        self.intrinsic = open3d.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fx, self.cx, self.cy)

    def getOdometry(self, img, disparity):
        self.prevLeft = self.curLeft
        self.prevDisparity = self.curDisparity

        self.curLeft = img
        self.curDisparity = disparity
        
        option = open3d.OdometryOption()
        option.max_depth = 10
        option.iteration_number_per_pyramid_level = open3d.IntVector([20, 20, 20])
        
        odo_init = np.identity(4)

        if self.prevLeft is not None:
            prev_rgbd = open3d.create_rgbd_image_from_color_and_depth(open3d.Image(self.prevLeft), open3d.Image(self.prevDisparity.astype(np.float32)), depth_scale=1000, depth_trunc=100)
            cur_rgbd = open3d.create_rgbd_image_from_color_and_depth(open3d.Image(self.curLeft), open3d.Image(self.curDisparity.astype(np.float32)), depth_scale=1000, depth_trunc=100)

            [success_hybrid_term, odometryEstimate, info] = open3d.compute_rgbd_odometry(prev_rgbd, cur_rgbd, self.intrinsic, self.prevOdometry, open3d.RGBDOdometryJacobianFromColorTerm(), option)

            #target_pcd = open3d.create_point_cloud_from_rgbd_image(cur_rgbd, self.intrinsic)
            #open3d.draw_geometries([target_pcd])

            if success_hybrid_term:
                print(odometryEstimate)
                
                self.prevOdometry = odometryEstimate
                
            return odometryEstimate
        return np.identity(4)

def saveImages(frameL, frameR):
    # check if main directory exists
    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)

    # check if left/right subdirectories exist
    if not os.path.exists(os.path.join(args.save_directory, "left")):
        os.mkdir(args.save_directory + "/left")
    if not os.path.exists(os.path.join(args.save_directory, "right")):
        os.mkdir(args.save_directory + "/right")
    
    # get current time to timestamp each picture
    currentTime = str(time.time())

    # write images to disk
    cv2.imwrite(os.path.join(args.save_directory, "left", currentTime + "_L.jpg"), frameL)
    cv2.imwrite(os.path.join(args.save_directory, "right", currentTime + "_R.jpg"), frameR)


parser = argparse.ArgumentParser(description="Simple reconstruction system for ZED stereo camera")
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-s", "--serial", type=int, help="camera serial number", default=0)
parser.add_argument("-cf", "--config_file", type=str, help="ZED camera calibration configuration file", default='')
parser.add_argument("-xml", "--config_file_xml", type=str, help="manual camera calibration XML configuration file", default='')
parser.add_argument("-fix", "--correct_focal_length", action='store_true', help="correct for error in VGA factory supplied focal lengths for earlier production ZED cameras")
parser.add_argument("-dir", "--save_directory", type=str, help="directory for captured images to be stored", default="Images")
parser.add_argument("-p", "--photos", action="store_true", help="use if need to record dataset")
parser.add_argument("-r", "--reconstruction", action="store_true", help="use reconstruction sequence on captured data")
args = parser.parse_args()


if __name__ == "__main__":
    ################################################################################
    # code from zed-stereo.py
    # process agruments to get camera calibration

    camera_calibration_available = False;
    manual_camera_calibration_available = False;

    if (args.serial > 0):

        url = 'http://calib.stereolabs.com/?SN=';

        # we have a serial number - go get the config file from the config url

        r = requests.get(url+str(args.serial));

        if ((r.status_code == requests.codes.ok) and not("ERROR" in r.text)):

            with open("zed-cam-sn-"+str(args.serial)+".conf", "w") as config_file:
                config_file.write(r.text[1:]); # write to file skipping first blank character

            path_to_config_file = "zed-cam-sn-"+str(args.serial)+".conf";

        else:
            print("Error - failed to retrieve camera config from: " + url);
            print();
            parser.print_help();
            exit(1);

        camera_calibration_available = True;

    elif (len(args.config_file) > 0):

        path_to_config_file = args.config_file;
        camera_calibration_available = True;

    elif (len(args.config_file_xml) > 0):

        path_to_config_file = args.config_file_xml;
        manual_camera_calibration_available = True;

    else:
        print("Warning - no serial number or config file specified.");
        print();

    ################################################################################

    # select config profiles based on image dimensions

        # MODE  FPS     Width x Height  Config File Option
        # 2.2K 	15 	    4416 x 1242     2K
        # 1080p 30 	    3840 x 1080     FHD
        # 720p 	60 	    2560 x 720      HD
        # WVGA 	100 	1344 x 376      VGA

    config_options_width = OrderedDict({4416: "2K", 3840: "FHD", 2560: "HD", 1344: "VGA"});
    config_options_height = OrderedDict({1242: "2K", 1080: "FHD", 720: "HD", 376: "VGA"});

    # define video capture object as a threaded video stream
    if args.photos:
        zed_cam = CameraVideoStream();
        zed_cam.open(args.camera_to_use);

        if (zed_cam.isOpened()):
            ret, frame = zed_cam.read();
        else:
            print("Error - selected camera #", args.camera_to_use, " : not found.");
            exit(1);

        height, width, channels = frame.shape;

        ################################################################################

        try:
            camera_mode = config_options_width[width];
        except KeyError:
            print("Error - selected camera #", args.camera_to_use,
            " : resolution does not match a known ZED configuration profile.");
            print()
            exit(1)

        print()
        print("ZED left/right resolution: ", int(width/2), " x ",  int(height))
        print("ZED mode: ", camera_mode)
        print()
        print("Keyboard Controls:")
        print("space \t - change camera mode")
        print("p \t - save left and right image")
        print("s \t - start/stop frame capture")
        print("x \t - exit")
        print()

        ################################################################################

        # process config to get camera calibration from calibration file
        # by parsing camera configuration as an INI format file

        if (camera_calibration_available):
            cam_calibration = configparser.ConfigParser();
            cam_calibration.read(path_to_config_file);
            fx, fy, B, Kl, Kr, distCoeffsL, distCoeffsR, R, T, Q = zed_camera_calibration(cam_calibration, camera_mode, width, height);

            # correct factory supplied values if specified

            if ((args.correct_focal_length) and (camera_mode == "VGA")):
                fx = fx / 2.0;
                fy = fy / 2.0;
                Q[0][3] =  -1 * (width / 4); Q[1][3] = -1 * (height / 2);
                Q[2][3] = fx; Q[3][3] = 0; # as Lcx == Rcx
        elif (manual_camera_calibration_available):
            fx, fy, B, Kl, Kr, distCoeffsL, distCoeffsR, R, T, Q = read_manual_calibration(path_to_config_file);
            # no correction needed here

        ################################################################################

    max_disparity = 128
    window_size = 9
    block_size = 9

    leftStereoMatcher = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities = max_disparity, # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=block_size,
            P1=8 * 3 * window_size ** 2,       # 8*number_of_image_channels*SADWindowSize*SADWindowSize
            P2=32 * 3 * window_size ** 2,      # 32*number_of_image_channels*SADWindowSize*SADWindowSize
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=1,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_HH4
    )

    # set up for WLS filter disparity enhancement
    rightStereoMatcher = cv2.ximgproc.createRightMatcher(leftStereoMatcher)
    wlsFilter = cv2.ximgproc.createDisparityWLSFilter(leftStereoMatcher)
    wlsFilter.setLambda(5000)
    wlsFilter.setSigmaColor(1.2)

    # display window
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    # video capture loop
    keep_processing = args.photos
    recording = False
    reconstruction = args.reconstruction

    while (keep_processing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # if video file successfully open then read frame

        if (zed_cam.isOpened()):
            ret, frame = zed_cam.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue
        else:
            continue

        # split single ZED frame into left an right
        frameL = frame[:, 0:int(width/2), :]
        frameR = frame[:, int(width/2):width, :]

        cv2.imshow("Camera", np.hstack((frameL, frameR)))

        if recording:
            saveImages(frameL, frameR)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # start the event loop - essential
        # wait 100ms or less depending on processing time taken (i.e. 1000ms / 10 fps = 100 ms)

        key = cv2.waitKey(max(2, 100 - int(math.ceil(stop_t)))) & 0xFF

        if (key == ord('x')):
            keep_processing = False

        elif (key == ord('s')):
            if not recording:
                recording = True
            else:
                recording = False
                keep_processing = False

        elif (key == ord('p')):
            saveImages(frameL, frameR)

        elif (key == ord(' ')):

            # cycle camera resolutions to get the next one on the list

            pos = 0
            list_widths = list(config_options_width.keys())
            list_heights = list(config_options_height.keys())

            list_widths.sort(reverse=True)
            list_heights.sort(reverse=True)

            print(list_widths)
            print(list_heights)
            
            for (width_resolution, config_name) in config_options_width.items():

                    if (list_widths[pos % len(list_widths)] == width):

                        camera_mode = config_options_width[list_widths[(pos-1) % len(list_widths)]]

                        # get new camera resolution

                        width = next(key for key, value in config_options_width.items() if value == camera_mode)
                        height = next(key for key, value in config_options_height.items() if value == camera_mode)

                        print ("Changing camera config to use: ", camera_mode, " @ ", width, " x ", height)
                        break

                    pos -= 1

            # reset to new camera resolution
            zed_cam.pause()
            zed_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            zed_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            zed_cam.resume()

            width = int(zed_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(zed_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print ("Camera config confirmed back from camera as: ", width , " x ", height)
            print()
            print("ZED left/right resolution: ", int(width/2), " x ",  int(height))
            print("ZED mode: ", camera_mode)
            print()

            # reset window sizes
            cv2.resizeWindow("Camera", width, height)
            
            # get calibration for new camera resolution

            if (camera_calibration_available):
                fx, fy, B, Kl, Kr, distCoeffsL, distCoeffsR, R, T, Q = zed_camera_calibration(cam_calibration, camera_mode, width, height)

            ####################################################################

    # release camera
    if args.photos:
        zed_cam.release()

    cv2.destroyAllWindows()

    if args.reconstruction:
        # output debug info from open3d
        open3d.set_verbosity_level(open3d.VerbosityLevel.Debug)
        
        # set up windows for reconstruction
        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

        # list filenames in order of timestamps
        leftImages = sorted(os.listdir(os.path.join(args.save_directory, "left")))
        rightImages = sorted(os.listdir(os.path.join(args.save_directory, "right")))      
        
        # read first left image
        image = cv2.imread(os.path.join(args.save_directory, "left", leftImages[0]), cv2.IMREAD_COLOR)
            
        # obtain image dimensions
        (height, width, channels) = image.shape

        # get camera calibration for the given resolution
        if args.photos:
            # makes further computations simpler
            Q[3][2] = -Q[3][2]
            odometry = Odometry(Q, height, width, fx, Kl[0][2], Kl[1][2])
        else:
            # retrieve calibration
            camera_mode = config_options_width[width * 2]

            # as in image capture
            if (camera_calibration_available):
                cam_calibration = configparser.ConfigParser()
                cam_calibration.read(path_to_config_file)
                fx, fy, B, Kl, Kr, distCoeffsL, distCoeffsR, R, T, Q = zed_camera_calibration(cam_calibration,  camera_mode, width, height)

                # correct factory supplied values if specified

                if ((args.correct_focal_length) and (camera_mode == "VGA")):
                    fx = fx / 2.0
                    fy = fy / 2.0
                    Q[0][3] =  -1 * (width / 4); Q[1][3] = -1 * (height / 2)
                    Q[2][3] = fx; Q[3][3] = 0; # as Lcx == Rcx
            elif (manual_camera_calibration_available):
                fx, fy, B, Kl, Kr, distCoeffsL, distCoeffsR, R, T, Q = read_manual_calibration(path_to_config_file)
                # no correction needed here

            Q[3][2] = -Q[3][2]
            odometry = Odometry(Q, height, width, fx, Kl[0][2], Kl[1][2])

        # initialise point cloud for reconstruction
        pcd = open3d.PointCloud()

        # iterate over images
        start = 0
        for filename in range(start, len(leftImages), 1):
            print("##############")
            print("Iteration:", filename + 1)
            print("##############")

            # read images
            leftImage = cv2.imread(os.path.join(args.save_directory, "left", leftImages[filename]), cv2.IMREAD_COLOR)
            rightImage = cv2.imread(os.path.join(args.save_directory, "right", rightImages[filename]), cv2.IMREAD_COLOR)

            # convert to grayscale
            leftGray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
            rightGray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

            leftGray = np.power(leftGray, 0.8).astype('uint8')
            rightGray = np.power(rightGray, 0.8).astype('uint8')

            # compute disparity
            leftDisp = np.int16(leftStereoMatcher.compute(leftGray, rightGray))
            rightDisp = np.int16(rightStereoMatcher.compute(rightGray, leftGray))
            disparityRaw = wlsFilter.filter(leftDisp, leftImage, None, rightDisp)

            disparity = cv2.threshold(disparityRaw, 0, max_disparity * 16, cv2.THRESH_TOZERO)[1] / 16.

            # change channel order to RGB for open3d processing
            imageRGB = cv2.cvtColor(leftImage, cv2.COLOR_BGR2RGB)

            # reproject points to 3D space and obtain depth image
            # points - 3D coordinates, colours - RGB values, depthImage - as disparity but in mm instead of px
            points, colours, depthImage = reproject_fast(imageRGB, disparity, Q)

            # compute odometry
            motionEstimate = odometry.getOdometry(imageRGB, depthImage)

            # transform running point cloud by motion estimate
            # motion estimate is inverse transformation in this case
            pcd.transform(motionEstimate)

            # create temporary point cloud for current frame
            tmp = open3d.PointCloud()
            
            # add points and colours (float in range [0..1]) to the temporary point cloud
            tmp.points = open3d.Vector3dVector(np.array(points))
            tmp.colors = open3d.Vector3dVector(np.array(colours) / 255.)

            # crop the point cloud to exclude distant points (>9.5 metres)
            # uses a cube to select points
            tmp = open3d.crop_point_cloud(tmp, np.array([-20.0000, -20.0000, -20.0000]), np.array([20., 20.0000, 9.5000]))

            # add points to running point cloud
            pcd += tmp

            # use voxel downsampling to align points to a grid 8mm in size
            # removes stitching artifacts and ensures uniform point density
            pcd = open3d.voxel_down_sample(pcd, 0.008)
            #open3d.write_point_cloud("reconstruction.ply", pcd)
            #open3d.draw_geometries([pcd])
            
            # display images
            cv2.imshow("Left", leftImage)
            cv2.imshow("Right", rightImage)
            cv2.imshow("Disparity", disparity.astype("uint8"))
            cv2.waitKey(1)

        # display the reconstruction
        open3d.draw_geometries([pcd])

        # writes point cloud to file, can be viewed in open3d or meshlab
        open3d.write_point_cloud("reconstruction2.ply", pcd)
