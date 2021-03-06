import cv2
import matplotlib.pyplot as plt
import numpy as np


def img_greying(r, g, b):
    return (r * 30 + g * 59 + b * 11) / 100


def img_darken(r, g, b, offset):
    return [(b - offset) if (b - offset) > 0 else 0,
            (g - offset) if (g - offset) > 0 else 0,
            (r - offset) if (r - offset) > 0 else 0]


def img_brighten(img, offset):
    return np.clip(img * 1. + offset, 0, 255)


def img_contrast(img, expand):
    return np.clip((img - 127.) * expand + 127., 0, 255)


def img_gamma(img, gamma):
    return np.power(img/255., 1/gamma) * 255.


def histogram_calculating(img):
    histogram = np.zeros([256, img.shape[2]])
    for g in range(0, 256):
        histogram[g, :] = sum(sum(img == g))
    return histogram


def cdf_calculating(histogram, img):
    cdf = np.zeros([256, img.shape[2]])
    tot = sum(histogram)
    for g in range(0, 256):
        cdf[g] = sum(histogram[0:g + 1]) / tot
    return cdf


def img_histogram_balance(img):
    histogram = histogram_calculating(img)
    cdf = cdf_calculating(histogram, img)
    (B, G, R) = cv2.split(img)
    return cv2.merge([(cdf[B] * 255)[:, :, 0], (cdf[G] * 255)[:, :, 1], (cdf[R] * 255)[:, :, 2]])


def img_histogram_matching(src, dst, cdf_src, cdf_dst):
    out = np.zeros(src.shape)
    mj = np.min(dst)
    Mj = np.max(dst)
    mi = np.min(src)
    Mi = np.max(src)
    gj = mj
    for gi in range(mi, Mi + 1):
        while (gj < 255 and cdf_src[gi] < 1) and cdf_dst[gj] < cdf_src[gi]:
            gj = gj + 1
        out = out + gj * (src == gi)
    return out


def multi_band_matching(src, dst):
    histogram = histogram_calculating(src)
    cdf_src = cdf_calculating(histogram, src)
    histogram = histogram_calculating(dst)
    cdf_dst = cdf_calculating(histogram, dst)
    (B, G, R) = cv2.split(src)
    (B1, G1, R1) = cv2.split(dst)
    return cv2.merge([img_histogram_matching(B, B1, cdf_src[:, 0], cdf_dst[:, 0]),
                      img_histogram_matching(G, G1, cdf_src[:, 1], cdf_dst[:, 1]),
                      img_histogram_matching(R, R1, cdf_src[:, 2], cdf_dst[:, 2])])


def main():
    img = cv2.imread("./qianxun.jpg")
    img_p = cv2.imread("./people.jpg")
    meizi = cv2.imread("./human.jpg")

    GreyImg = np.zeros(img.shape, dtype=np.int)
    BrightenImg = np.zeros(img.shape, dtype=np.int)
    row_cnt = -1
    col_cnt = -1
    for row in img:
        row_cnt += 1
        for pixel in row:
            col_cnt += 1
            GreyImg[row_cnt, col_cnt] = img_greying(pixel[2], pixel[1], pixel[0])
            BrightenImg[row_cnt, col_cnt] = img_darken(pixel[2], pixel[1], pixel[0], 50)
        col_cnt = -1

    cv2.imwrite("qianxun_grey.jpg", GreyImg)
    cv2.imwrite("qianxun_darken.jpg", BrightenImg)
    cv2.imwrite("qianxun_contrast.jpg", img_contrast(img, 1.5))
    cv2.imwrite("qianxun_brighten.jpg", img_brighten(img, 50))
    cv2.imwrite("qianxun_gamma.jpg", img_gamma(img, 1.5))
    cv2.imwrite("people_equlization.jpg", img_histogram_balance(img_p))
    cv2.imwrite("people_matching.jpg", multi_band_matching(meizi, img))


if __name__ == '__main__':
    main()
