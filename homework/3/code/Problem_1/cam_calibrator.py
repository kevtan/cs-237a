#!/usr/bin/env python

import itertools
import os
import pdb
import sys

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from camera_calibration.calibrator import (ChessboardInfo, MonoCalibrator,
                                           Patterns)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        cols, rows = n_corners
        self.boards.append(ChessboardInfo(cols, rows, float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' +
                             file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(
                img, 'mono8')         # Convert to ROS Image msg
            # Extract chessboard corners using ROS camera_calibration package
            drawable = self.c.handle_msg(img_msg)

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98,
                                    top=0.98, bottom=0.02)
                fig.canvas.set_window_title(
                    'Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        # Length of a chessboard square
        self.d_square = square_length
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        # Number of examined images
        self.n_chessboards = len(self.c.good_corners)
        # Dimensions of extracted corner grid
        self.n_corners_y, self.n_corners_x = n_corners
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

    def genCornerCoordinates(self, u_meas, v_meas):
        '''
        Inputs:
            u_meas: a list of arrays where each array are the u values for each board.
            v_meas: a list of arrays where each array are the v values for each board.
        Output:
            corner_coordinates: a tuple (Xg, Yg) where Xg/Yg is a list of arrays where
            each array are the x/y values for each board.

        HINT: u_meas, v_meas starts at the blue end, and finishes with the pink end
        HINT: our solution does not use the u_meas and v_meas values
        HINT: it does not matter where your frame it, as long as you are consistent!
        '''
        EPSILON = 0.001
        x_coords = np.arange(0, self.n_corners_x *
                             self.d_square - EPSILON, self.d_square)
        y_coords = np.arange(0, self.n_corners_y *
                             self.d_square - EPSILON, self.d_square)
        X_board = np.zeros(self.n_corners_per_chessboard)
        Y_board = X_board.copy()
        for i, coords in enumerate(itertools.product(y_coords, x_coords)):
            Y_board[i], X_board[i] = coords
        X_boards = tuple(itertools.repeat(X_board, self.n_chessboards))
        Y_boards = tuple(itertools.repeat(Y_board, self.n_chessboards))
        return X_boards, Y_boards

    def estimateHomography(self, u_meas, v_meas, X, Y):    # Zhang Appendix A
        '''
        Given the pixel coordinates of points on the model plane and their associated
        world coordinates, return an estimate of the homography matrix H.

        Inputs:
            u_meas: an array of the u values for a board.
            v_meas: an array of the v values for a board.
            X: an array of the X values for a board. (from genCornerCoordinates)
            Y: an array of the Y values for a board. (from genCornerCoordinates)
        Output:
            H: the homography matrix. its size is 3x3

        HINT: What is the size of the matrix L?
        HINT: What are the outputs of the np.linalg.svd function? Based on this, where does the eigenvector corresponding to the smallest eigen value live?
        HINT: np.stack and/or np.hstack may come in handy here.
        '''
        # find the homogeneous world coordinates for every point
        M_tildes = [np.array([x, y, 1]) for x, y in zip(X, Y)]

        def constraints(M_tilde, u, v):
            """Returns a tuple of 2 arrays that are rows in the constraint matrix."""
            zeros = np.zeros(3)
            row1 = np.concatenate((M_tilde, zeros, -u*M_tilde))
            row2 = np.concatenate((zeros, M_tilde, -v*M_tilde))
            return row1, row2
        rows = map(lambda triple: constraints(
            *triple), zip(M_tildes, u_meas, v_meas))
        L = np.array(list(itertools.chain.from_iterable(rows)))
        # compute the SVD of constraint matrix L
        u, s, vh = np.linalg.svd(L)
        H = vh[np.argmin(s)].reshape((3, 3))
        return H

    def getCameraIntrinsics(self, H):    # Zhang 3.1, Appendix B
        '''
        Input:
            H: a list of homography matrices for each board
        Output:
            A: the camera intrinsic matrix

        HINT: MAKE SURE YOU READ SECTION 3.1 THOROUGHLY!!! V. IMPORTANT
        HINT: What is the definition of h_ij?
        HINT: It might be cleaner to write an inner function (a function inside the getCameraIntrinsics function)
        HINT: What is the size of V?
        '''
        def constraints(H):
            """Returns a tuple of 2 arrays that are rows in the constraint matrix."""
            h1, h2 = H[:, 0], H[:, 1]

            def construct_v(h_a, h_b):
                """Constructs vector v_ab"""
                h_a1, h_a2, h_a3 = h_a
                h_b1, h_b2, h_b3 = h_b
                return np.array([
                    h_a1*h_b1,
                    h_a1*h_b2+h_a2*h_b1,
                    h_a2*h_b2,
                    h_a3*h_b1+h_a1*h_b3,
                    h_a3*h_b2+h_a2*h_b3,
                    h_a3*h_b3
                ])
            v12 = construct_v(h1, h2)
            v11 = construct_v(h1, h1)
            v22 = construct_v(h2, h2)
            return v12, (v11-v22)
        V = np.array(list(itertools.chain.from_iterable(map(constraints, H))))
        u, s, vh = np.linalg.svd(V)
        B11, B12, B22, B13, B23, B33 = vh[np.argmin(s)]
        # calculate intrinsic camera parameters
        v_0 = (B12*B13-B11*B23)/(B11*B22-B12**2)
        lamb = B33 - ((B13**2+v_0*(B12*B13-B11*B23))/B11)
        alpha = np.sqrt(lamb/B11)
        beta = np.sqrt((lamb*B11)/(B11*B22-B12**2))
        gamma = (-B12*alpha**2*beta)/lamb
        u_0 = gamma*v_0/beta-B13*alpha**2/lamb
        return np.array([
            [alpha, gamma, u_0],
            [0, beta, v_0],
            [0, 0, 1]
        ])

    def getExtrinsics(self, H, A):
        '''
        Inputs:
            H: a single homography matrix
            A: the camera intrinsic matrix
        Outputs:
            R: the rotation matrix
            t: the translation vector

        Note: Zhang 3.1, Appendix C
        '''
        # extract the columns of the homography matrix
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        # precompute constants
        A_inv = np.linalg.inv(A)
        lamb = 1/np.linalg.norm(np.matmul(A_inv, h1))
        # calculate rotation matrix columns
        r1 = lamb*np.matmul(A_inv, h1)
        r2 = lamb*np.matmul(A_inv, h2)
        r3 = np.cross(r1, r2)
        # assemble rotation matrix columns
        R = np.concatenate(
            tuple(map(lambda col: col.reshape(3, 1), [r1, r2, r3])), axis=1)
        # find a true rotation matrix most similar to R
        u, s, vh = np.linalg.svd(R)
        R_approx = np.matmul(u, vh)
        # calculate translation vector
        t = lamb*np.matmul(A_inv, h3)
        return R_approx, t

    # Zhang 2.1, Eq. (1)
    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x, y: the coordinates in the ideal normalized image plane

        '''
        t = t.reshape((3, 1))
        transformation = np.concatenate((R, t), axis=1)

        def perform_mapping(X_coord, Y_coord, Z_coord):
            """Maps a point in world coordinates into a camera coordinates."""
            M_tilde = np.array([X_coord, Y_coord, Z_coord, 1])
            x_coord, y_coord, w = np.matmul(transformation, M_tilde)
            return x_coord/w, y_coord/w
        x, y = np.zeros_like(X), np.zeros_like(Y)
        for i, (x_coord, y_coord) in enumerate(map(perform_mapping, X, Y, Z)):
            x[i], y[i] = x_coord, y_coord
        return x, y

    # Zhang 2.1, Eq. (1)
    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u, v: the coordinates in the ideal pixel image plane
        '''
        x, y = self.transformWorld2NormImageUndist(X, Y, Z, R, t)

        def perform_mapping(x_coord, y_coord):
            """Maps a point in camera coordinates into pixel coordinates."""
            m_tilde = np.array([x_coord, y_coord, 1])
            u_coord, v_coord, w = np.matmul(A, m_tilde)
            return u_coord/w, v_coord/w
        u, v = np.zeros_like(x), np.zeros_like(y)
        for i, (u_coord, v_coord) in enumerate(map(perform_mapping, x, y)):
            u[i], v[i] = u_coord, v_coord
        return u, v

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure(
            'Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(
                X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height *
                             0.15, box.width, box.height*0.85])
            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
                      fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x *
                       self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(
                    np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0]
                              [1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title(
                    'Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(
            ([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            # Flip Y-axis to traditional direction
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])

        return u_meas, v_meas   # Lists of arrays (one per chessboard)
