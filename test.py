import argparse
import cv2
import joblib
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import preprocessing


def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    if (train is True):
        np.random.shuffle(images)

    return images


def draw_keypoints(img, kp):
    keypoints = cv2.drawKeypoints(img, kp, outImage=None)
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints)
    plt.show()


def getDescriptors(sift, img, feature_extraction):
    if feature_extraction == 'kp':
        kp, des = sift.detectAndCompute(img, None)
        # draw_keypoints(img, kp)
        return des
    if feature_extraction == 'grid':
        step_size = 10
        kp = [cv2.KeyPoint(x, y, step_size) for y in
              range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]

        kp, des = sift.compute(img, kp)
        # draw_keypoints(img, kp)

        return des


def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))
    # return img


def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors


def clusterDescriptors(descriptors, no_clusters, alg):
    if alg == 'kmeans':
        kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
        return kmeans
    elif alg == 'meanshift':

        descriptors_normalized = preprocessing.scale(descriptors)
        bandwidth = estimate_bandwidth(descriptors_normalized, quantile=0.3,
                                       n_samples=500)

        ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)
        ms.fit(descriptors_normalized)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)

        return ms


def extractFeatures(cluster, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = cluster.predict(feature)
            im_features[i][idx] += 1

    return im_features


def normalizeFeatures(scale, features):
    return scale.transform(features)


def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array(
        [abs(np.sum(im_features[:, h], dtype=np.int32)) for h in
         range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
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


def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = ["airplanes", "cars", "faces", "motorbikes"]
    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()


def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))


def trainModel(path, no_clusters, clustering_alg, feature_extraction):
    images = getFiles(True, path)
    print("Train images path detected.")
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 7
    image_count = len(images)

    for img_path in images:
        if ("airplanes" in img_path):
            class_index = 0
        elif ("cars" in img_path):
            class_index = 1
        elif ("faces" in img_path):
            class_index = 2
        elif ("motorbikes" in img_path):
            class_index = 3

        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img, feature_extraction)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    cluster = clusterDescriptors(descriptors, no_clusters, alg=clustering_alg)
    print("Descriptors clustered with " + clustering_alg)

    im_features = extractFeatures(cluster, descriptor_list, image_count,
                                  no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    clf = LinearSVC()
    clf.fit(im_features, np.array(train_labels))
    print("SVM fitted.")
    print("Training completed.")

    return cluster, scale, clf, im_features


def testModel(path, cluster, scale, svm, im_features, no_clusters,
              feature_extraction):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    name_dict = {
        "0": "airplanes",
        "1": "cars",
        "2": "faces",
        "3": "motorbikes",
    }

    sift = cv2.xfeatures2d.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img, feature_extraction)

        if (des is not None):
            count += 1
            descriptor_list.append(des)

            if ("airplanes" in img_path):
                true.append("airplanes")
            elif ("cars" in img_path):
                true.append("cars")
            elif ("faces" in img_path):
                true.append("faces")
            elif ("motorbikes" in img_path):
                true.append("motorbikes")

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(cluster, descriptor_list, count,
                                    no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(true, predictions)
    print("Confusion matrixes plotted.")

    findAccuracy(true, predictions)
    print("Accuracy calculated.")
    print("Execution done.")


def execute(train_path, test_path, no_clusters, clustering_alg,
            feature_extraction, save_file):
    cluster, scale, svm, im_features = trainModel(train_path, no_clusters,
                                                 clustering_alg, feature_extraction)


    # Save the SVM
    joblib.dump((cluster, scale, svm, im_features), (save_file + ".pkl"),
                compress=3)

    cluster, scale, svm, im_features = joblib.load(save_file + ".pkl")

    testModel(test_path, cluster, scale, svm, im_features, no_clusters, feature_extraction)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path",
                        required=True)
    parser.add_argument('--test_path', action="store", dest="test_path",
                        required=True)
    parser.add_argument('--no_clusters', action="store", dest="no_clusters",
                        default=50)
    parser.add_argument('--clustering_alg', action="store",
                        dest="clustering_alg", default="cluster", required=True)
    parser.add_argument('--feature_extraction', action="store",
                        dest="feature_extraction", default="kp",
                        required=True)
    parser.add_argument('--save_file', action="store",
                        dest="save_file", default="latest")
    args = vars(parser.parse_args())

    execute(args['train_path'], args['test_path'], int(args['no_clusters']),
            args['clustering_alg'], args['feature_extraction'],
            args['save_file'])
