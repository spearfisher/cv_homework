import cv2
import numpy as np
import os

OUTPUT_PATH = "results3/"
K_CLUSTERS = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH_RIVER = os.path.join(SCRIPT_DIR, "dataset", "fields_and_river")
SOURCE_PATH_ROADS = os.path.join(SCRIPT_DIR, "dataset", "fields_and_roads")
SOURCE_PATH_CITIES = os.path.join(SCRIPT_DIR, "dataset", "fields_and_cities")

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
files = [f for f in os.listdir(SOURCE_PATH_RIVER) if f.endswith(('.png', '.jpg', '.jpeg'))]

def process_image(img_path):
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # blur = cv2.medianBlur(img, 7)

    data = blur.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    bright_cluster = np.argmax(np.mean(centers, axis=1))
    mask_bright = (labels.reshape(img.shape[:2]) == bright_cluster).astype(np.uint8) * 255

    idx = np.where(mask_bright > 0)
    only_bright_view = np.zeros_like(img)
    only_bright_view[idx] = img[idx]
    cv2.imshow('What K-means sees now', only_bright_view)
    cv2.waitKey(0)

for filename in files:
    process_image(os.path.join(SOURCE_PATH_RIVER, filename))
