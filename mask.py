#Cite: https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv

import numpy as np
import cv2

limits = [([0, 120, 0], [140, 255, 100]), ([25, 0, 75], [180, 38, 255])]

def masking_operation(frame):

    mask_x = cv2.inRange(frame, np.array(limits[0][0], dtype="uint8"), np.array(limits[0][1], dtype="uint8"))
    mask_y = cv2.inRange(frame, np.array(limits[1][0], dtype="uint8"), np.array(limits[1][1], dtype="uint8"))

    mask_xy = cv2.bitwise_or(mask_x, mask_y)
    masked = cv2.bitwise_and(frame, frame, mask=mask_xy)

    return masked