# -*-coding=utf-8-*-
import numpy as np
import math

class Graph:
    def __init__(self, vertex_list=[], tri_list=[]):
        self.vertex_list = vertex_list
        self.tri_list = tri_list
    
    def add_element(self, ops, element):
        if ops == 'vertex':
            self.vertex_list.append(element)
        elif ops == 'triangle':
            self.tri_list.append(element)
    
    def del_repeated_element(edge_list):
        N = len(edge_list)
        filter = np.zeros(N)
        deduplicated_list = []
        for i in range(0, N):
            if filter[i] == 0:
                for j in range(i + 1, N):
                    if edge_list[i].equals(edge_list[j]):
                        filter[i] = 1
                        filter[j] = 1
        for i in range(0, N):
            if filter[i] == 0:
                deduplicated_list.append(edge_list[i])
        return deduplicated_list


class Triangle:
    def __init__(self, pts=[]):
        self.pts = pts
        [pt1, pt2, pt3] = pts
        [x1, y1] = pt1
        [x2, y2] = pt2
        [x3, y3] = pt3
        a = x1 - x2
        b = y1 - y2
        c = x1 - x3
        d = y1 - y3
        e = (pow(x1, 2) - pow(x2, 2) - pow(y2, 2) + pow(y1, 2)) / 2
        f = (pow(x1, 2) - pow(x3, 2) - pow(y3, 2) + pow(y1, 2)) / 2
        self.x0 = (b*f - d*e) / (b*c - a*d)
        self.y0 = (c*e - a*f) / (b*c - a*d)
        self.r =  self.dist((x1, y1), (self.x0, self.y0))

    def in_circle(self, pts):
        distance = self.dist((self.x0, self.y0), pts)
        if distance < self.r:
            return True
        return False

    def dist(self, pt1, pt2):
        [x1, y1] = pt1
        [x2, y2] = pt2
        return math.sqrt(pow(abs(x1-x2), 2) + pow(abs(y1-y2), 2))

    def overlap(self, pts):
        for inner_pt in self.pts:
            for outter_pt in pts:
                if inner_pt == outter_pt or inner_pt == (outter_pt[1], outter_pt[0]):
                    return True
        return False


class Edge:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pts = [pt1, pt2]
    
    def equals(self, edge):
        if edge.pt1 == self.pt1 and edge.pt2 == self.pt2:
            return True
        elif edge.pt2 == self.pt1 and edge.pt1 == self.pt2:
            return True
        return False