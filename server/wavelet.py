import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.abs(imArray_H)
    imArray_H = imArray_H / imArray_H.max()
    imArray_H = np.uint8(imArray_H * 255)

    return imArray_H