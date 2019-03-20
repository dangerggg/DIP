
import face_recognition
import cv2


def face_keypoint_recognition(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    return face_landmarks_list


