import cv2
import numpy as np


def photo_replant(src_img, dst_img, loc_list):
    img = np.zeros(dst_img)
    img[loc_list[0]:loc_list[1], loc_list[2]:loc_list[3]] = src_img
    return img


def possion_img_processing(img, loc_region):
    return img


