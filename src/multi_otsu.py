import numpy as np

import os
import sys

from skimage.filters import threshold_multiotsu
from skimage import io

image_path = sys.argv[1]

#Reading Image Height and Width
image = io.imread(image_path,0)
print(f'Image Shape: {image.shape}')
image_height, image_width = image.shape

#Obtain thresholds
thresholds = threshold_multiotsu(image, 4) #4 classes 

#Segment based on thresholds
regions = np.digitize(image, bins=thresholds)

#initializing new image using the zeros
new_image = np.zeros((image_height, image_width, 3), np.uint8)

pores_index = regions==0
new_image[pores_index] = np.array([0,0,255], np.uint8) #Blue

clay_index = regions==1
new_image[clay_index] = [0,255,0] #Green

quartz_index = regions ==2
new_image[quartz_index] = [255,0,0] #Red

heavy_index = regions ==3
new_image[heavy_index] = [255,255,0] #Yellow

image_name = image_path.split(os.sep)[-1][:-4]
output_name =os.path.join('results',image_name+'_otsu_mask.png')

# io.imsave("../results/otsu_output_mask.png", new_image)

io.imsave(output_name, new_image)