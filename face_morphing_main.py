import point_processing as pp
import face_detection as face

import cv2


def region_coloring(loc_list, img):
    for loc in loc_list:
        img[loc[1] - 1:loc[1] + 1, loc[0] - 1:loc[0] + 1] = (255, 255, 255)
    return img


def main():
    img = cv2.imread("girl.jpg")
    face_marks = face.face_keypoint_recognition(img)
    for key in face_marks[0]:
       img = region_coloring(face_marks[0][key], img)
    cv2.imwrite("people_recognition.jpg", img)


if __name__ == "__main__":
    main()
