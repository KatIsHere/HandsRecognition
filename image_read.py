import cv2
import glob
import numpy as np
from scipy import misc

    #  load images path
imgLeftHand_TrainPath = "Training_LeftHand50x50/*.jpg"
imgRightHand_TrainPath = "Training_RightHand50x50/*.jpg"
imgBad_TrainPath = "Training_Bad50x50/*.jpg"
imgLeftHand_TestPath = "Testing_LeftHand50x50/27731/*.jpg"
imgRightHand_TestPath = "Testing_RightHand50x50/27730/*.jpg"
imgBad_TestPath = "Testing_Bad50x50/27733/*.jpg"
    
    # image classes
_CLASES_ = ["rightHand", "leftHand", "notHand", "reversedRight", "reversedLeft"]

def load_images(filepath, blackAndWhite = True):
    """Loads set of images from a folder to an array"""
    if blackAndWhite:
        return [cv2.imread(file, 0) for file in glob.glob(filepath)]
    else:
        return [cv2.imread(file) for file in glob.glob(filepath)]

def get_reversed(img):
    hight, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((hight/2, width/2), 180, 1.0)
    imgRotated = cv2.warpAffine(img, M, (hight, width))
    return imgRotated

def classLen():
    return len(_CLASES_)

def read_data():
    """Returns datasets af numpy arrays"""
    num_classes = len(_CLASES_)

    # image -> np.array
    train_LeftImg =load_images(imgLeftHand_TrainPath)
    test_LeftImg = load_images(imgLeftHand_TestPath)
    train_RightImg = load_images(imgRightHand_TrainPath)
    test_RightImg = load_images(imgRightHand_TestPath)
    train_BadImg = load_images(imgBad_TrainPath)
    test_BadImg = load_images(imgBad_TestPath)

    # get reversed classes
    test_LeftImg_Reversed = np.array([get_reversed(img) for img in test_LeftImg])
    test_RightImg_Reversed = np.array([get_reversed(img) for img in test_RightImg])
    train_LeftImg_Reversed = np.array([get_reversed(img) for img in train_LeftImg])
    train_RightImg_Reversed = np.array([get_reversed(img) for img in train_RightImg])

    # classes
    train_numb = np.array([len(train_RightImg), len(train_LeftImg), len(train_BadImg), \
                            len(train_RightImg_Reversed), len(train_LeftImg_Reversed)])
    test_numb = np.array([len(test_RightImg), len(test_LeftImg), len(test_BadImg), \
                            len(test_RightImg_Reversed), len(test_LeftImg_Reversed)])

    # making a set of training/testing images and their labels
    train_data = np.concatenate((train_RightImg, train_LeftImg, train_BadImg,   \
                    train_RightImg_Reversed, train_LeftImg_Reversed), axis = 0)
    test_data = np.concatenate((test_RightImg, test_LeftImg, test_BadImg,       \
                    test_RightImg_Reversed, test_LeftImg_Reversed), axis = 0)
    train_labels = np.concatenate([[i for num in range(train_numb[i])] for i in range(num_classes)])
    test_labels = np.concatenate([[i for num in range(test_numb[i])] for i in range(num_classes)])

    return (train_data, train_labels), (test_data, test_labels)
