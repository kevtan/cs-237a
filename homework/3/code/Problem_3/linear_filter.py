#!/usr/bin/env python

import itertools
import pdb
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    (Irows, Icols, _), (Frows, Fcols, _) = I.shape, F.shape
    padded = np.pad(I, ((0, Frows), (0, Fcols), (0, 0)), "constant")
    G = np.empty((Irows, Icols))
    # pre-vectorize the filter
    f = F.flatten()
    for i, j in itertools.product(range(Irows), range(Icols)):
        # vectorize the image region
        t_ij = padded[i:i+Frows, j:j+Fcols].flatten()
        G[i, j] = np.dot(f, t_ij)
    return G


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    (Irows, Icols, _), (Frows, Fcols, _) = I.shape, F.shape
    padded = np.pad(I, ((0, Frows), (0, Fcols), (0, 0)), "constant")
    G = np.empty((Irows, Icols))
    # pre-vectorize the filter
    f = F.flatten()
    f_mag = np.linalg.norm(f)
    for i, j in itertools.product(range(Irows), range(Icols)):
        # vectorize the image region
        t_ij = padded[i:i+Frows, j:j+Fcols].flatten()
        t_ij_mag = np.linalg.norm(t_ij)
        G[i, j] = np.dot(f, t_ij) / (f_mag * t_ij_mag)
    return G


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = norm_cross_corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
