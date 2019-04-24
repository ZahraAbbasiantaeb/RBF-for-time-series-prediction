import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans


def create_sin_data():

    training_data = pd.read_csv("/Users/zahra_abasiyan/PycharmProjects/Project/deep_learning_course/datasets/HW02_RBF_Dataset1_daily-total-female-births-in-cal.csv")

    data = training_data.as_matrix()

    size = np.shape(data)[0]

    train_x = []

    train_y = []

    kmeans_data = []


    for index in range(0, size):

        kmeans_data.append((index, data[index, 1]))

        train_x.append(index)

        train_y.append(data[index, 1])

    return train_x, train_y, kmeans_data


train_x, train_y, kmeans_data = create_sin_data()

cluster_count = 360

cls = KMeans(cluster_count)

cls.fit(kmeans_data)

labels = cls.labels_

sigmas = np.zeros(cluster_count)

centers = cls.cluster_centers_[:,0]

train_size =  np.shape(train_x)[0]

dimension = 1


def plot_scatter(X, y):

    plt.scatter(X, y)

    plt.show()

    return


def plot_scatter_two(x, y1, y2):

    plt.scatter(x, y1)

    plt.scatter(x, y2)

    plt.show()

    return


def get_phi_x(X, center, sigma):

    return math.exp(np.linalg.norm([np.subtract(X, center)] ,ord=2) / (-2*sigma))


def getSigmaRadial():

    dist = []

    for center in centers:

        for center2 in centers:
            dist.append(np.sqrt(np.sum(np.subtract(center2, center) ** 2)))

    sigma = max(dist) / np.sqrt(2 * cluster_count)

    return sigma


def getHiddenInput(train_x):

    train_size=np.shape(train_x)[0]

    input = np.zeros((train_size, cluster_count))

    sigma = getSigmaRadial()

    for i in range(0, train_size):

        for j in range(0, cluster_count):

            input[i,j] = get_phi_x(train_x[i], centers[j], sigma)

    return input


def getHiddenInputEbf(train_x):

    train_size=np.shape(train_x)[0]

    input = np.zeros((train_size, cluster_count))

    sigma = getSigmaRadial()

    for i in range(0, train_size):

        for j in range(0, cluster_count):

            input[i,j] = get_phi_ebf_x(train_x[i], centers[j], sigma)

    return input


def get_phi_ebf_x(x, center, cov):

    return math.exp((distance.mahalanobis(x, center, np.transpose(cov)) ** 2) / (-2))


def setCovarianceOfEachCluster():

    cov_matrix = []

    for index in set(labels):

        count = 0
        tmp = np.zeros((dimension, dimension))
        i = 0
        tmp = np.matrix(tmp)
        for num in labels:

            if num == index:
                count += 1
                mat_tmp = np.matrix(train_x[i] - centers[index])
                mat =  np.transpose(mat_tmp) @ mat_tmp
                tmp += mat

            i += 1

        tmp /= count
        cov_matrix.append(tmp)

    return cov_matrix

train_rbf_x = getHiddenInput(train_x)

train_ebf_x = getHiddenInputEbf(train_x)

plt.scatter(train_x, train_y)

plt.show()

print(train_y)

