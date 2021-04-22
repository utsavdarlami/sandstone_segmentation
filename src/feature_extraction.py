import numpy as np
import cv2
# import os
# import pandas as pd
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import sobel, roberts, scharr, prewitt
from scipy import ndimage as nd

def gabor_feature_extractor(image):
    """
    Applies 48 gabor filters and
    returns the dictionary containing
    the features obatined after applying on those images
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


def all_feature_extractor(imgpath):
    """
        Applies 58 filters and
        returns the dictionary containing
        the features obatined after applying on those images
    """
    
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Extracting Gabor Features
    feature_dict = gabor_feature_extractor(image)
    
    
    feature_dict['Original'] = image
    
    entropy_img = entropy(image, disk(1))
    feature_dict['Entropy'] = entropy_img
    
    gaussian3_img = nd.gaussian_filter(image, sigma=3)
    feature_dict['Gaussian3'] = gaussian3_img
    
    gaussian7_img = nd.gaussian_filter(image, sigma = 7)
    feature_dict['Gaussian7'] = gaussian7_img
    
    sobel_img = sobel(image)
    feature_dict['Sobel'] = sobel_img
    
    canny_edge_img = cv2.Canny(image,100,200)
    feature_dict['Canny'] = canny_edge_img
    
    robert_edge_img  = roberts(image)
    feature_dict['Robert'] = robert_edge_img
    
    scharr_edge = scharr(image)
    feature_dict['Scharr'] =scharr_edge
    
    prewitt_edge = prewitt(image)
    feature_dict['Prewitt'] = prewitt_edge
    
    median_img = nd.median_filter(image, size = 3)
    feature_dict['Median'] = median_img
    
    variance_img = nd.generic_filter(image, np.var, size = 3)
    feature_dict['Variance'] = variance_img
    
    return feature_dict