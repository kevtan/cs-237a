#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import pdb


def template_match(template, image, threshold=0.999):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    correlation = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    _, thresholded = cv2.threshold(correlation, threshold, 1, cv2.THRESH_BINARY)
    X, Y = np.nonzero(thresholded)
    r, c, _ = template.shape
    return [(x, y, r, c) for x, y in zip(X, Y)]


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

    cv2.imwrite(filename, 255*det_img/det_img.max())


def main():
    image = cv2.imread('clutter.png').astype(np.float32)
    template = cv2.imread('valdo.png').astype(np.float32)

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, threshold=0.85)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == "__main__":
    main()
