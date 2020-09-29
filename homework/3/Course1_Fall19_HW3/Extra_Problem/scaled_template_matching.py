#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools

import template_matching as tm


def template_match(template, image, nUpscales=2, nDownscales=3, threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped template ndarray
        image: An (m, n, c)-shaped image ndarray
        nUpscales: times to 2x-upscale image with Gaussian blur
        nUpscales: times to 0.5x-downscale image with Gaussian blur
        threshold: min normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (TL_y, TL_x, H, W) tuples for bounding boxes
    """
    # construct image pyramid
    pyramid = [None] * nDownscales + [(1.0, image)] + [None] * nUpscales
    for i in range(nDownscales-1, -1, -1):
        prev_scale, prev_image = pyramid[i+1]
        pyramid[i] = (prev_scale/2, cv2.pyrDown(prev_image))
    for j in range(nDownscales+1, nDownscales+nUpscales+1):
        prev_scale, prev_image = pyramid[j-1]
        pyramid[j] = (prev_scale*2, cv2.pyrUp(prev_image))
    # perform template matching
    def match_and_scale(pyramid_entry):
        scale, image = pyramid_entry
        boxes = tm.template_match(template, image, threshold)
        return map(lambda box: map(lambda dim: int(dim/scale), box), boxes)
    return list(itertools.chain.from_iterable(map(match_and_scale, pyramid)))


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    # template = cv2.imread('messi_face.jpg')
    # image = cv2.imread('messipyr.jpg')

    # matches = template_match(template, image)
    # create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()
