from prepare_dataset import get_single_image_dataframe
import sys
import cv2
import pickle
import numpy as np

if __name__ == "__main__":

    # get image path as argument
    input_image_path = sys.argv[1]

    # create dataframe of all the features
    image_feature = get_single_image_dataframe(input_image_path)

    image = cv2.imread(input_image_path, 0)
    image_height = image.shape[0]
    image_width = image.shape[1]

    # loading the model
    model_path = './models/dtree.pkl'
    loaded_model = pickle.load(open(model_path, 'rb'))

    # predicting the labels for a single feature
    pred_labels = loaded_model.predict(image_feature)

    segmented_image = pred_labels.reshape((image_height, image_width))

    """ Coloring the segmented path """
    new_image = np.zeros((image_height, image_width, 3),
                         np.uint8)

    pores_index = segmented_image == 29
    new_image[pores_index] = np.array([0, 0, 255],
                                      np.uint8)  # blue

    quartz_index = segmented_image == 76
    new_image[quartz_index] = [255, 0, 0]  # red

    clay_index = segmented_image == 150
    new_image[clay_index] = [0, 255, 0]  # green

    heavy_index = segmented_image == 226
    new_image[heavy_index] = [255, 255, 0]  # yellow

    image_name = input_image_path.split("/")[-1][:-4]
    RGB_mask = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    output_name = "./results/" + image_name + "_mask.png"
    print(f"Saving the mask as {output_name}")
    cv2.imwrite(output_name, RGB_mask)
