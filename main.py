import os

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

def custom_listdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def train_model(path):
    images = custom_listdir(path)
    return images

def get_images(path):
    imgs = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            imgs.append(os.path.join(root, f))

    return imgs


if __name__ == '__main__':
    img = cv2.imread('dataset/airplanes_train/img005.jpg')

    train_path = 'dataset-modified/train'
    test_path = 'dataset-modified/test'

    # Get the training classes names and store them in a list
    images = get_images(train_path)
    train_labels = np.array([])

    for path in images:
        if ("airplanes" in path):
            class_index = 0
        elif ("cars" in path):
            class_index = 1
        elif ("motorbikes" in path):
            class_index = 2
        elif ("faces" in path):
            class_index = 3

        train_labels = np.append(train_labels, class_index)

    a=1






    # des_grid = get_descriptors_grid(img, step_size=10)
    des_key = get_descriptors_default(img)

