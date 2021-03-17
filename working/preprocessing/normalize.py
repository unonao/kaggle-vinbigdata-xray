import numpy as np
import cv2

def normalize_xray(img, way="hist"):
    """
    x線の画像(配列)を正規化する
    way:
        hist: ヒストグラム正規化
        clahe: Contrast Limited AHE 、
    """
    if way == "hist":
        norm_img = cv2.equalizeHist(img)
        return norm_img
    elif way == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm_img = clahe.apply(img)
        return norm_img
    elif way == "original":
        return img
    else:
        raise("no such way")