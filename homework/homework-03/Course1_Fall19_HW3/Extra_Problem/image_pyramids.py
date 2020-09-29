#!/usr/bin/env python

import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Down samples an image by selecting only even-indexed rows and columns.

    Parameters:
    - image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns:
    - downscaled_image: A half-downscaled version of image.
    """
    rows, cols = map(lambda n: int(n/2+0.5), image.shape[:-1])
    downscaled = np.zeros((rows, cols, 3))
    for r, c in itertools.product(range(rows), range(cols)):
        downscaled[r, c] = image[r*2, c*2]
    return downscaled


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.7)
    return half_downscale(blurred)


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape
    f = (1./scale) * np.convolve(np.ones((scale,)), np.ones((scale,)))
    f = np.expand_dims(f, axis=0) # Making it (1, scale)-shaped
    filt = f.T * f
    upscaled = np.zeros((m*scale, n*scale, c))
    for r, c in itertools.product(range(m), range(n)):
        upscaled[r*scale, c*scale] = image[r, c]
    return cv2.filter2D(upscaled, -1, filt)


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png').astype(float)#[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png').astype(float)#[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    # downsize1 = half_downscale(test_card)
    # downsize2 = half_downscale(downsize1)
    # downsize3 = half_downscale(downsize2)
    # cv2.imwrite("downsize1.png", 255*downsize1/downsize1.max())
    # cv2.imwrite("downsize2.png", 255*downsize2/downsize2.max())
    # cv2.imwrite("downsize3.png", 255*downsize3/downsize3.max())
    # downsize1 = blur_half_downscale(test_card)
    # downsize2 = blur_half_downscale(downsize1)
    # downsize3 = blur_half_downscale(downsize2)
    # cv2.imwrite("blur_downsize1.png", 255*downsize1/downsize1.max())
    # cv2.imwrite("blur_downsize2.png", 255*downsize2/downsize2.max())
    # cv2.imwrite("blur_downsize3.png", 255*downsize3/downsize3.max())

    # upscale = two_upscale(two_upscale(two_upscale(favicon)))
    # cv2.imwrite("two_upscale.png", 255*upscale/upscale.max())
    upscale = bilinterp_upscale(favicon, 8)
    cv2.imwrite("bilinear_upscale.png", 255*upscale/upscale.max())

if __name__ == '__main__':
    main()
