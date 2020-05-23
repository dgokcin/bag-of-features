import numpy as np
import cv2
import pylab as plt


def get_descriptors_default(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    draw_keypoints(gray, kp)

    return des


def get_descriptors_grid(img, step_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = [cv2.KeyPoint(x, y, step_size) for y in
          range(0, gray.shape[0], step_size)
          for x in range(0, gray.shape[1], step_size)]

    kp, des = sift.compute(gray, kp)
    draw_keypoints(gray, kp)

    return des


def draw_keypoints(img, kp):
    keypoints = cv2.drawKeypoints(img, kp, outImage=None)
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('dataset/airplanes_train/img005.jpg')

    des_grid = get_descriptors_grid(img, step_size=10)
    des_key = get_descriptors_default(img)

