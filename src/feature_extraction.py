import numpy as np
import cv2
# import os
# import pandas as pd


def gabor_feature_extractor(image):
    """
    Applies 48 gabor filters and
    returns the dictionary containing
    the features obatined after applying those images
    """

    ksize = 9
    phi = 0
    gabor_features = {}
    count = 0
    for i in range(3):  # 0, 45, and 90 degree
        theta = i/4.0 * np.pi
        for sigma in (1, 3):
            # lam 0 to 135 degree with 45degree step
            for lam in np.arange(0.1, np.pi, np.pi/4.0):
                for gamma in (0.05, 0.5):
                    gabor_name = "gabor_" + str(count)
                    kernel = cv2.getGaborKernel((ksize, ksize),
                                                sigma,
                                                theta,
                                                lam,
                                                gamma,
                                                phi,
                                                ktype=cv2.CV_32F)
                    feature = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    gabor_features[gabor_name] = feature
                    count += 1

    return gabor_features
