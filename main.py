import os

import joblib
import numpy as np
import cv2
import pylab as plt
import argparse as ap
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


def get_descriptors_default(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    # draw_keypoints(gray, kp)

    return kp, des


def get_descriptors_grid(img, step_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = [cv2.KeyPoint(x, y, step_size) for y in
          range(0, gray.shape[0], step_size)
          for x in range(0, gray.shape[1], step_size)]

    kp, des = sift.compute(gray, kp)
    # draw_keypoints(gray, kp)

    return kp, des


def draw_keypoints(img, kp):
    keypoints = cv2.drawKeypoints(img, kp, outImage=None)
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints)
    plt.show()


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if (f.endswith(
        '.jpg') or f.endswith('.png'))]


def get_images(path):
    imgs = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            imgs.append(os.path.join(root, f))

    return imgs

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = ["airplanes", "cars", "faces", "motorbikes"]
    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()


def plotConfusionMatrix(y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))

if __name__ == '__main__':
    train_path = 'dataset-modified/train'

    # Get the training classes names and store them in a list
    training_names = mylistdir(train_path)
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imlist(dir)
        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        class_id += 1

    print(image_classes)

    # List where all the descriptors are stored
    des_list = []

    for path in image_paths:
        img = cv2.imread(path)
        # des_grid_1 = get_descriptors_grid(img, step_size=10)
        # des_grid_2 = get_descriptors_grid(img, step_size=10)
        kpts, des = get_descriptors_default(img)
        des_list.append((path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Perform k-means clustering
    k = 150
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    print(im_features)

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(
        np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)),
        'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Plot Histogram
    plotHistogram(im_features, k)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))

    # Save the SVM
    joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)

    ########################################################################

    clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")
    test_path = 'dataset-modified/test'

    testing_names = mylistdir(test_path)
    image_paths = []

    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imlist(dir)
        image_paths += class_path

    des_list = []

    test_labels = []
    for path in image_paths:
        img = cv2.imread(path)
        kpts, des = get_descriptors_default(img)
        des_list.append((path, des))
        # Fill test labels array for future use
        test_labels.append(path.split('/')[2])

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    print(descriptors)
    #
    test_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
    idf = np.array(
        np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)),
        'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)
    print(test_features)

    # Perform the predictions
    predictions = [classes_names[i] for i in clf.predict(test_features)]
    findAccuracy(test_labels, predictions)
    plotConfusions(test_labels, predictions)
    a = 1




