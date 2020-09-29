#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *
import itertools
import pdb


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r, y_r, theta, rho = RangeData

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-splitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split, INCLUSIVE.
        endIdx: ending index of segment to be split, EXCLUSIVE.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.
    '''
    alpha, r = FitLine(theta[startIdx:endIdx], rho[startIdx:endIdx])
    num_points = endIdx - startIdx
    if num_points <= params["MIN_POINTS_PER_SEGMENT"]:
        return np.array([alpha]), np.array([r]), np.array([[startIdx, endIdx]])
    split = FindSplit(theta[startIdx:endIdx], rho[startIdx:endIdx], alpha, r, params)
    if split == -1:
        return np.array([alpha]), np.array([r]), np.array([[startIdx, endIdx]])
    alpha1, r1, i1 = SplitLinesRecursive(theta, rho, startIdx, startIdx+split, params)
    alpha2, r2, i2 = SplitLinesRecursive(theta, rho, startIdx+split, endIdx, params)
    alphas = np.concatenate((alpha1, alpha2))
    rs = np.concatenate((r1, r2))
    indices = np.concatenate((i1, i2))
    return alphas, rs, indices

def FindSplit(thetas, rhos, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    if 2.01 < r < 2.02:
        pdb.set_trace()
    distances = np.zeros_like(thetas)
    for i, (theta, rho) in enumerate(zip(thetas, rhos)):
        distances[i] = np.abs((rho * np.cos(theta-alpha)) - r)
    max_indx = np.argmax(distances)
    max_dist = distances[max_indx]
    if max_dist > params["LINE_POINT_DIST_THRESHOLD"]:
        threshold = params["MIN_POINTS_PER_SEGMENT"]
        nPointsSeg1 = max_indx
        nPointsSeg2 = len(thetas) - max_indx
        if nPointsSeg1 >= threshold and nPointsSeg2 >= threshold:
            return max_indx
    return -1

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads).
        r: 'r' of best fit for range data (1 number) (m).
    '''
    n = len(theta)
    A = sum(r**2 * np.sin(2.*t) for r, t in zip(rho, theta))
    B = sum(r * np.cos(t) for r, t in zip(rho, theta))
    C = sum(r * np.sin(t) for r, t in zip(rho, theta))
    numerator = A - (2. / n) * B * C
    D = sum(r**2 * np.cos(2.*t) for r, t in zip(rho, theta))
    rhos = itertools.product(rho, rho)
    thetas = itertools.product(theta, theta)
    E = sum(r1*r2*np.cos(t1+t2) for (r1, r2), (t1, t2) in zip(rhos, thetas))
    denominator = D - (1. / n) * E
    alpha  = 0.5 * np.arctan2(numerator, denominator) + np.pi/2.
    r = (1./n) * sum(r * np.cos(t - alpha) for r, t in zip(rho, theta))
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    alphaOut, rOut, pointIdxOut = list(alpha), list(r), list(pointIdx)
    for i in range(len(alpha)-1, 0, -1):
        a, _ = pointIdxOut[i-1]
        _, b = pointIdxOut[i]
        alpha, r = FitLine(theta[a:b], rho[a:b])
        s = FindSplit(theta[a:b], rho[a:b], alpha, r, params)
        if s == -1:
            # cannot find split: merge
            alphaOut.pop(i)
            alphaOut.pop(i-1)
            alphaOut.insert(i-1, alpha)
            rOut.pop(i)
            rOut.pop(i-1)
            rOut.insert(i-1, r)
            pointIdxOut.pop(i)
            pointIdxOut.pop(i-1)
            pointIdxOut.insert(i-1, [a, b])
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 3  # minimum number of points per line segment
    MAX_P2P_DIST = 0.4  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    # filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
