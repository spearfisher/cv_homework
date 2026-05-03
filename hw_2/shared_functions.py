import glob
import os

import cv2 as cv


def loadImagesPaths(directory):
    image_paths = sorted(glob.glob(os.path.join(directory, '*.png')))
    if not image_paths:
        raise FileNotFoundError(f'No images found in {directory}')
    return image_paths


def readImage(image_path, flags=cv.IMREAD_COLOR):
    img = cv.imread(image_path, flags)
    if img is None:
        raise FileNotFoundError(f'Image at {image_path} could not be loaded.')
    return img
