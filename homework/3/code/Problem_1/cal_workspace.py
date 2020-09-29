#!/usr/bin/env python

import pdb  # TODO remove

import matplotlib.pyplot as plt
import numpy as np

from cam_calibrator import CameraCalibrator


def main():
    # create a camera calibrator
    cc = CameraCalibrator()

    cal_img_path = './astra_23'  # Location of calibration images
    name = 'astra'               # Name of the camera
    n_corners = [7, 9]           # Corner grid dimensions
    square_length = 0.0205       # Chessboard square length in meters

    # WORKFLOW PARAMETERS
    # n_disp_img: Number of images for each plot to display (1 to 23)
    #             Feel free to change this to 10 (or fewer) while developing/debugging
    #             so that you don't have to click through 23 images for each step
    # display_flag: True to display images and corners as they are loaded
    #               Feel free to change this to False while developing/debugging
    n_disp_img = 10  # 23
    display_flag = True

    cc.loadImages(cal_img_path, name, n_corners,
                  square_length, n_disp_img, display_flag)

    # find the pixel image coordinates via feature extraction
    u_meas, v_meas = cc.getMeasuredPixImageCoord()
    X, Y = cc.genCornerCoordinates(u_meas, v_meas)

    # estimate homographies for each image
    H = [cc.estimateHomography(u, v, x, y)
         for u, v, x, y in zip(u_meas, v_meas, X, Y)]

    # use estimated homographies to determine true camera intrinsic parameters
    A = cc.getCameraIntrinsics(H)

    R = []
    t = []
    for p in range(cc.n_chessboards):
        Rout, tout = cc.getExtrinsics(H[p], A)
        R.append(Rout)
        t.append(tout)

    cc.plotBoardPixImages(u_meas, v_meas, X, Y, R, t, A, n_disp_img)

    cc.plotBoardLocations(X, Y, R, t, n_disp_img)

    k = [0.15, 0.01]

    cc.plotBoardPixImages(u_meas, v_meas, X, Y, R, t, A, n_disp_img, k)

    cc.undistortImages(A, k, n_disp_img)

    cc.writeCalibrationYaml(A, k)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
