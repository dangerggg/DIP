# -*-coding=utf-8-*-
import POINT_PROCESSING as pp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import lil_matrix
from scipy.linalg import solve

def read_img(path):
    return cv2.imread(path)

def write_img(img, path):
    cv2.imwrite(path, img)

def display_img(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img) 
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def spliter(processed_img):  
    segment = []
    [col, r, depth] = processed_img.shape
    for c in range(0, col):
        for row in range(0, r):
            pixel = processed_img[c, row]
            if (pixel != [0, 0, 0]).all():
                segment.append((c, row))
    return segment

def adjacent_pixel(img, col, r):
    [b_col, b_r] = img.shape
    if col == 0 and r == 0:
        return [(col+1, r), (col, r+1)]
    elif 0 < col < b_col-1 and r == 0:
        return [(col-1, r), (col+1, r), (col, r+1)]
    elif col == b_col-1 and r == 0:
        return [(col-1, r), (col, r+1)]
    elif col == b_col-1 and 0 < r < b_r-1:
        return [(col-1, r), (col, r-1), (col, r+1)]
    elif col == b_col-1 and r == b_r-1:
        return [(col-1, r), (col, r-1)]
    elif 0 < col < b_col-1 and r == b_r-1:
        return [(col-1, r), (col+1, r), (col, r-1)]
    elif col == 0 and r == b_r-1:
        return [(col+1, r), (col, r-1)]
    elif col == 0 and 0 < r < b_r-1:
        return [(col+1, r), (col, r-1), (col, r+1)]
    else:
        return [(col+1, r), (col-1, r), (col, r-1), (col, r+1)]

def compute_gradient(img):
    gradient_xy = []
    gradient_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #Prewitt kernel
    gradient_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) #Prewitt kernel
    gradient_xy.append(cv2.filter2D(img, -1, gradient_kernel_x))
    gradient_xy.append(cv2.filter2D(img, -1, gradient_kernel_y))
    return gradient_xy

def compute_laplacian(img):
    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])      #Laplace kernel
    return cv2.filter2D(img, -1, laplace_kernel)

def border_detection(col, r, mask): #Refering to a blog, sites: https://alexfxw.github.io/2019/04/14/PoissonEditing/
    if mask[col, r] == 0:
        return False
    for i, j in adjacent_pixel(mask, col, r):
        if mask[i, j] == 0:
            return True
    return False

def possion_solver(src, target, segment, mask):  #Refering to a CSDN blog, constructing Ax = b 
    N = len(segment)                             #Then solve it with opencv libs
    A = lil_matrix((N, N))                       #sites: https://blog.csdn.net/hjimce/article/details/45716603
    b = np.zeros(N)
    print(A.shape)
    #-----------------get b according to mask---------------------#
    laplacian = compute_laplacian(src)
    for loc in range(0, N):
        [col, r] = segment[loc]
        b[loc] = laplacian[col, r]
        #-----------------considering border condition---------------------#
        # Refering to: https://blog.csdn.net/hjimce/article/details/45716603
        if border_detection(col, r, mask) == True:
            for i, j in adjacent_pixel(mask, col, r):
                if mask[i, j] == 0:
                    b[loc] += target[i, j]
    #-----------------get A according to mask---------------------#
    for loc in range(0, N):
        A[loc, loc] = 4
        [col, r] = segment[loc]
        for point in adjacent_pixel(mask, col, r):
            if point in segment:
                A[loc, segment.index(point)] = -1
    x = sparse_alg.spsolve(A, b)
    print(x)
    return x

def merge_img(src, target, segment, mask):
    #------------raw version of merging-------------------#
    raw_target = np.zeros(target.shape)
    for point in segment:
        pixel = src[point[0], point[1]]
        raw_target[point[0], point[1]] = pixel
    # commented blocks are used to get the gradient of the image
    # while some bugs occured when solving Ax = b, so I just give up the solve joined by gradient
    """gradient_src = compute_gradient(src)   
    gradient_target = compute_gradient(target)
    for point in segment: #masking...
        pixel_x = gradient_src[0][point[0], point[1]]
        pixel_y = gradient_src[1][point[0], point[1]]
        gradient_target[0][point[0], point[1]] = pixel_x
        gradient_target[1][point[0], point[1]] = pixel_y"""
   #------------possion image editing below--------------#
    x = possion_solver(src, target, segment, mask)
    result = np.copy(target).astype(int)
    N = len(segment)
    for loc in range(0, N):
        [col, r] = segment[loc]
        result[col, r] = x[loc]
    #----------point processing of image below------------#
    return result

def given_offset(src, mask, target, offset=(0, 0)): # If the mask is not aligned with the target image, we need to shift it... But how to shift it is important
    resize = target.shape
    new_src = np.zeros(resize)
    new_mask = np.zeros(resize)
    [offset_col, offset_r] = offset
    [col, r, depth] = src.shape
    [new_col, new_r, new_depth] = resize
    for c in range(0, col):
        for row in range(0, r):
            if 0 <= c+offset_col < new_col and 0 <= row+offset_r < new_r:       
                new_mask[c+offset_col, row+offset_r] = mask[c, row]
                new_src[c+offset_col, row+offset_r] = src[c, row]
    return [new_src, new_mask]


def main():
    offset_x = 160#-134
    offset_y = 140#-89
    mask1 = read_img("../image fusion/mask.jpg")
    src1 = read_img("../image fusion/source.jpg")
    target1 = read_img("../image fusion/target.jpg")
    [src1, mask1] = given_offset(src1, mask1, target1, (0, 0))
    #write_img(mask1, "../image fusion/merge1.jpg")
    [src_b1, src_g1, src_r1] = cv2.split(src1)
    [target_b1, target_g1, target_r1] = cv2.split(target1)
    [mask_b1, mask_g1, mask_r1] = cv2.split(mask1)
    segment1 = spliter(mask1)
    merge1 = cv2.merge([merge_img(src_b1, target_b1, segment1, mask_b1),
                       merge_img(src_g1, target_g1, segment1, mask_g1),
                       merge_img(src_r1, target_r1, segment1, mask_r1)])
    write_img(merge1, "../image fusion/merge3.jpg")
    #display_img(merge1)

    """mask2 = read_img("../image fusion/test2_mask.png")
    src2 = read_img("../image fusion/test2_src.png")
    target2 = read_img("../image fusion/test2_target.png")
    segment2 = spliter(mask1)
    merge2 = merge_img(src2, target2, segment2, 160, 140, mask2)
    #display_img(merge2)"""


if __name__ == '__main__':
    main()
