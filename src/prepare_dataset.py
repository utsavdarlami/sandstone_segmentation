from feature_extraction import all_feature_extractor
import os
import cv2
import pandas as pd
import sys


def get_single_image_dataframe(imagepath):
    """
    Creating a single dataframe by reshaping
    the features in the all_feature_dict
    """

    single_image_dataframe = pd.DataFrame()

    all_feature_dict = all_feature_extractor(imagepath)
    for i, feature in enumerate(all_feature_dict):
        single_image_dataframe[feature] = all_feature_dict[feature].reshape(-1)

    return single_image_dataframe


if __name__ == "__main__":

    no_of_images = int(sys.argv[1])
    print(f"Preparing Dataset with {no_of_images} images")

    BASE_DIR = os.getcwd()
    data_dir = BASE_DIR + "/data/raw/"
    images_path = data_dir + "Train_images/"
    masks_path = data_dir + "Train_masks/"
    images = sorted(os.listdir(images_path))[:no_of_images]
    masks = sorted(os.listdir(masks_path))[:no_of_images]
    final_dataframe = pd.DataFrame()

    for image, mask in zip(images, masks):
        print(f"- Extracting the features from {image}")

        image_path = os.path.join(images_path, image)
        single_image_dataframe = get_single_image_dataframe(image_path)

        """
        Add Mask label
        """
        mask_path = os.path.join(masks_path, mask)
        read_mask = cv2.imread(mask_path)
        read_mask_gray = cv2.cvtColor(read_mask, cv2.COLOR_BGR2GRAY)
        single_image_dataframe["Mask_label"] = read_mask_gray.reshape(-1)

        # Concat the single dataframe with other image df
        final_dataframe = pd.concat([final_dataframe, single_image_dataframe],
                                    ignore_index=True,
                                    axis=0,
                                    sort=False)

        print(f"- Done Extracting the features from {image}")
        # break

    print("- Saving the dataframe as final_dataset.csv is ../data/processed/")
    print(f"- Shape {final_dataframe.shape}")
    print(f"- Columns {final_dataframe.columns}")

    dataset_name = "./data/processed/final_dataset.csv"
    final_dataframe.to_csv(dataset_name, index=False)

    print("- Done preparing the dataset")
