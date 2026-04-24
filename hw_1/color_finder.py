import cv2
import numpy as np
import os

from shared_functions import readImage

def nothing(x):
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
img_raw = readImage(os.path.join(script_dir, 'dataset/kpi/kpi_bing.png'))

scale_percent = 50
width = int(img_raw.shape[1] * scale_percent / 100)
height = int(img_raw.shape[0] * scale_percent / 100)
img = cv2.resize(img_raw, (width, height))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Interface')

cv2.createTrackbar('Low H', 'Interface', 35, 179, nothing)
cv2.createTrackbar('High H', 'Interface', 85, 179, nothing)
cv2.createTrackbar('Low S', 'Interface', 20, 255, nothing)
cv2.createTrackbar('High S', 'Interface', 150, 255, nothing)
cv2.createTrackbar('Low V', 'Interface', 20, 255, nothing)
cv2.createTrackbar('High V', 'Interface', 255, 255, nothing)

while True:
    l_h = cv2.getTrackbarPos('Low H', 'Interface')
    h_h = cv2.getTrackbarPos('High H', 'Interface')
    l_s = cv2.getTrackbarPos('Low S', 'Interface')
    h_s = cv2.getTrackbarPos('High S', 'Interface')
    l_v = cv2.getTrackbarPos('Low V', 'Interface')
    h_v = cv2.getTrackbarPos('High V', 'Interface')

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([h_h, h_s, h_v])

    mask = cv2.inRange(hsv, lower, upper)
    
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    res = cv2.bitwise_and(img, img, mask=mask)
    combined = np.hstack((img, mask_3ch, res))

    cv2.imshow('Interface', combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()