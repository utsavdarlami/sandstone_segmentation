import sys
import cv2
import numpy as np
from sklearn.metrics import jaccard_score, f1_score

if __name__ == "__main__":
    
    # get obtained mask image path as argument
    obtained_mask_path = sys.argv[1]
    # get obtained mask image path as argument
    real_mask_path = sys.argv[2]

    obtained_mask = cv2.imread(obtained_mask_path)
    real_mask = cv2.imread(real_mask_path)

    obtained_mask_gray = cv2.cvtColor(obtained_mask, cv2.COLOR_BGR2GRAY)
    real_mask_gray = cv2.cvtColor(real_mask, cv2.COLOR_BGR2GRAY)

    real_mask_gray_reshaped = real_mask_gray.reshape(-1)  # y_test
    obtained_mask_gray_reshaped = obtained_mask_gray.reshape(-1)  # y_pred

    iou_score = jaccard_score(real_mask_gray_reshaped, obtained_mask_gray_reshaped, average=None)

    labels = [29, 76, 150, 226]

    print("The IOU Scores of predicted segmented mask:")
    for score, label in zip(iou_score, labels):
        print(f"For {label}, the score is: {score:.2f}")


    print()

    print("The Dice Scores of predicted segmented mask:")
    dice_score = f1_score(real_mask_gray_reshaped, obtained_mask_gray_reshaped, average=None)

    for score, label in zip(dice_score, labels):
        print(f"For{label}, the score is: {score:.2f}")



    





