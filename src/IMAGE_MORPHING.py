# -*-coding=utf-8-*-
import IMAGE_FUSION as basic_io
import Graph as graph
import numpy as np
from scipy.linalg import solve
import cv2
import dlib
import face_recognition
import math
import copy

def facial_points_detection(src, target): # only useful for both images are human face
    face_landmarks_src = face_recognition.face_landmarks(src)
    face_landmarks_target = face_recognition.face_landmarks(target)
    face_landmarks_src_list = []
    face_landmarks_target_list = []
    for key in face_landmarks_src[0]:
        face_landmarks_src_list += face_landmarks_src[0][key]
    for key in face_landmarks_target[0]:
        face_landmarks_target_list += face_landmarks_target[0][key]
    N = len(face_landmarks_src_list)
    for loc in range(0, N):
        face_landmarks_src_list[loc] = (face_landmarks_src_list[loc][1], face_landmarks_src_list[loc][0])
        face_landmarks_target_list[loc] = (face_landmarks_target_list[loc][1], face_landmarks_target_list[loc][0])
    return [face_landmarks_src_list, face_landmarks_target_list]

def show_facial_points(src, points): # print facial points on raw image
    show_src = np.array(src)
    for pixel in points:
        [col, r] = pixel
        show_src[col, r] = [255, 0, 0]
    basic_io.display_img(show_src)

def draw_line(pt1, pt2, img):
    if pt1[0] == pt2[0]:
        for y in range(min(pt1[1], pt2[1])+1, max(pt1[1], pt2[1])):
            img[pt1[0], y] = [0, 0, 255]
    else:
        for x in range(min(pt1[0], pt2[0])+1, max(pt1[0], pt2[0])):
            k = (pt1[1]-pt2[1]) / (pt1[0]-pt2[0])
            y = int(k * (x-pt1[0])) + pt1[1]
            img[x, y] = [0, 0, 255]
    return img

def draw_triangulation(tri_list, img):
    [col, r, depth] = img.shape
    for tri in tri_list:
        [pt1, pt2, pt3] = tri.pts
        img = draw_line(pt1, pt2, img)
        img = draw_line(pt1, pt3, img)
        img = draw_line(pt2, pt3, img)
    basic_io.write_img(img, "../image morphing/triangulation.jpg")

def transform_facial_points(src, target, alpha):
    N = len(src)
    intermediate = []
    assert N == len(target)
    assert 0 <= alpha <= 1
    for loc in range(0, N):
        intermediate.append((int(alpha*src[loc][0] + (1.0-alpha)*target[loc][0]),int(alpha*src[loc][1] + (1.0-alpha)*target[loc][1])))
    return intermediate

def subroutine_triangulate_strategy(pts, height, width):# 
    hypo_pts = [(-height, -width), (-height, 3*width), (3*height, -width)]
    hypo_triangle = graph.Triangle(hypo_pts)
    tri_list = []
    vertex_list = pts
    img_graph = graph.Graph(vertex_list, tri_list)
    # use area-adding to solve the delaulay triangulation
    tri_list.append(hypo_triangle)
    for pt in vertex_list:
        edge_list = []
        N = len(tri_list)
        for loc in range(N-1, -1, -1):
            if tri_list[loc].in_circle(pt) == True:
                [pt1, pt2, pt3] = tri_list[loc].pts
                edge_list.append(graph.Edge(pt1, pt2))
                edge_list.append(graph.Edge(pt2, pt3))
                edge_list.append(graph.Edge(pt1, pt3))
                del(tri_list[loc])
                
        edge_list = graph.Graph.del_repeated_element(edge_list)     
        for edge in edge_list:
            tri_pts = [pt, edge.pt1, edge.pt2]
            a = pt[0] - edge.pt1[0]
            b = pt[1] - edge.pt1[1]
            c = pt[0] - edge.pt2[0]
            d = pt[1] - edge.pt2[1]
            if b*c != a*d:
                tri_list.append(graph.Triangle(tri_pts))
    N = len(tri_list)
    for loc in range(N-1, -1, -1):
        if tri_list[loc].overlap(hypo_pts) == True:
            del(tri_list[loc])
    return tri_list

def delauny_triangulation(pts, height, width):
    return subroutine_triangulate_strategy(pts, height, width)

def solve_transform(src, target):
    [(x1, y1), (x2, y2), (x3, y3)] = src
    [(_x1, _y1), (_x2, _y2), (_x3, _y3)] = target
    b = np.array([_x1, _y1, _x2, _y2, _x3, _y3])
    A = np.array([[x1, y1, 1, 0, 0, 0],
                  [0, 0, 0, x1, y1, 1],
                  [x2, y2, 1, 0, 0, 0],
                  [0, 0, 0, x2, y2, 1],
                  [x3, y3, 1, 0, 0, 0],
                  [0, 0, 0, x3, y3, 1]])
    x = solve(A, b)
    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [0, 0, 1]])

def distance(pt1, pt2):
    [x1, y1] = pt1
    [x2, y2] = pt2
    return math.sqrt(pow(abs(x1-x2), 2) + pow(abs(y1-y2), 2))

def get_all_pts(triangle):
    [pt1, pt2, pt3] = triangle.pts
    pts_list = []
    radius = max([distance(pt1, pt2), distance(pt1, pt3), distance(pt2, pt3)])
    a = 1.0 / radius
    for p in np.arange(0., 1., a):
        for q in np.arange(0., 1-p, a):
            p_x = int(p*pt1[0] + q*pt2[0] + (1 - p - q)*pt3[0])
            p_y = int(p*pt1[1] + q*pt2[1] + (1 - p - q)*pt3[1])
            pts_list.append((p_x, p_y))
    return list(set(pts_list))

def merging(facial_points_src, facial_points_target, source, target, height, width, name):
    for alpha in np.arange(0.1, 1, 0.2):
        intermediate = transform_facial_points(facial_points_src, facial_points_target, 1 - alpha)
        tri = delauny_triangulation(intermediate, height, width)
        new_img = np.zeros([height, width, 3])
        for triangle in tri:
            [pt1, pt2, pt3] = triangle.pts
            index1 = intermediate.index(pt1)
            index2 = intermediate.index(pt2)
            index3 = intermediate.index(pt3)
            pt1_src2 = facial_points_src[index1]
            pt2_src2 = facial_points_src[index2]
            pt3_src2 = facial_points_src[index3]
            pt1_target2 = facial_points_target[index1]
            pt2_target2 = facial_points_target[index2]
            pt3_target2 = facial_points_target[index3]
            #--------------------------warping triangles------------------------------#
            T1 = solve_transform(triangle.pts, [pt1_src2, pt2_src2, pt3_src2])
            T2 = solve_transform(triangle.pts, [pt1_target2, pt2_target2, pt3_target2])
            pts = get_all_pts(triangle)
            for pt in pts:
                intermediate_src = np.dot(T1, np.array([pt[0], pt[1], 1]))
                intermediate_target = np.dot(T2, np.array([pt[0], pt[1], 1]))
                #print(intermediate_target)
                new_img[pt[0], pt[1]] = np.array((1.- alpha) * source[int(intermediate_src[0]), int(intermediate_src[1])] + alpha * target[int(intermediate_target[0]), int(intermediate_target[1])], dtype=int)
        basic_io.write_img(new_img, "../image morphing/" + name + "_" + str(alpha) + ".png")
        print("finish: " + str(alpha))

def main():
    #-----------------data preparation & initialization---------------------------------#
    facial_points_src2 = [(172, 60), (156, 95), (174, 127), (185, 81), (186, 102),
    (179, 183), (157, 218), (168, 248), (188, 214), (187, 229),
    (183, 155), (276, 156), (259, 120), (259, 190), (300, 100),
    (301, 157), (302, 204), (324, 157), (290, 53), (373, 125),
    (374, 182), (304, 255), (134, 34), (22, 79), (13, 217),
    (114, 282)]
    facial_points_target2 = [(159, 40), (144, 64), (174, 82), (166, 56), (172, 69),
    (167, 194), (133, 206), (141, 234), (158, 209), (150, 220),
    (162, 136), (274, 149), (273, 95), (260, 200), (376, 96),
    (337, 158), (354, 235), (364, 161), (273, 30), (410, 120),
    (400, 210), (245, 262), (169, 6), (58, 67), (45, 200),
    (125, 270)]
    source1 = basic_io.read_img("../image morphing/source1.png")
    target1 = basic_io.read_img("../image morphing/target1.png")
    source2 = basic_io.read_img("../image morphing/source2.png")
    target2 = basic_io.read_img("../image morphing/target2.png") 
    print(source1.shape)
    print(target1.shape)
    [height1, width1, depth1] = source1.shape
    [height2, width2, depth2] = source2.shape
    [height1_, width1_, depth1_] = target1.shape
    [height2_, width2_, depth2_] = target2.shape
    [facial_points_src1, facial_points_target1] = facial_points_detection(source1, target1)
    #-------------------------delauny triangulation-------------------------------------#
    facial_points_src1 += [(0, 0), (0, int(width1/2)), (0, width1-1), (int(height1/2), 0), 
    (int(height1/2), width1-1), (height1-1, 0), (height1-1, int(width1/2)), (height1-1, width1-1)]

    facial_points_src2 += [(0, 0), (0, int(width2/2)), (0, width2-1), (int(height2/2), 0), 
    (int(height2/2), width2-1), (height2-1, 0), (height2-1, int(width2/2)), (height2-1, width2-1)]

    facial_points_target1 += [(0, 0), (0, int(width1_/2)), (0, width1_-1), (int(height1_/2), 0), 
    (int(height1_/2), width1_-1), (height1_-1, 0), (height1_-1, int(width1_/2)), (height1_-1, width1_-1)]

    facial_points_target2 += [(0, 0), (0, int(width2_/2)), (0, width2_-1), (int(height2_/2), 0), 
    (int(height2_/2), width2_-1), (height2_-1, 0), (height2_-1, int(width2_/2)), (height2_-1, width2_-1)]
    #---------------merging two images, including computing affine matrix----------------#
    #merging(facial_points_src1, facial_points_target1, source1, target1, max(height1, height1_), max(width1, width1_), "intermediate1")
    tri = delauny_triangulation(facial_points_src2, height2, width2)
    draw_triangulation(tri, source2)
if __name__ == "__main__":
    main()
