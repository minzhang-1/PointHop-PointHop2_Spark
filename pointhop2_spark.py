import os
import time
import numpy as np
import modelnet_data
from pyspark import SparkConf
from numpy import linalg as LA
import point_utils_spark as pus
from pyspark import SparkContext
from sklearn.metrics import accuracy_score

config = SparkConf().setAll(
    [('spark.driver.memory', '14g'),
     ('spark.executor.memory', '8g'),
     ('spark.driver.maxResultSize', '14g')]).setAppName('PCSEG').setMaster('local[*]')
sc = SparkContext(conf=config)
sc.setLogLevel("ERROR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pointhop_train(data, n_newpoint, n_sample, threshold, num_partition):
    '''
    Train based on the provided samples.
    :param data: [num_samples, num_point, feature_dimension]
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param n_kernel: num kernels to be preserved
    :return: pca_params, feature
    '''

    point_data = data
    fea = []
    pca_params = {}
    pointRDD = sc.parallelize(point_data, num_partition)

    for i in range(len(n_sample)):
        if (i == 0 and point_data.shape[1] == n_newpoint[i]) or (i > 0 and n_newpoint[i-1] == n_newpoint[i]):
            fpsRDD = pointRDD
        else:
            fpsRDD = pointRDD.map(lambda x: pus.fps(x, n_newpoint[i]))
        fpsRDD.persist()

        knnRDD = fpsRDD.zip(pointRDD).map(lambda x: pus.knn(x[0], x[1], n_sample[i]))
        knnRDD.persist()

        if i == 0:
            sgRDD = pointRDD.zip(knnRDD).flatMap(lambda x: pus.sg(x[0], x[0], x[1]))
            sgRDD.persist()
            kernels, energy = pus.pca(sgRDD)
            pcaRDD = sgRDD.map(lambda x: np.dot(x, kernels.T))
            num_node = np.sum(energy > threshold)
            pre_energy = energy[:num_node]
            pca_fea = np.array(pcaRDD.collect())
            pca_fea = pca_fea.reshape((-1, n_newpoint[i], pca_fea.shape[-1]))
            pcaRDD = sc.parallelize(pca_fea[:, :, :num_node], num_partition)
            pca_leaf_fea = pca_fea[:, :, num_node:]
            print('Hop ', i, ': ', pca_fea[:, :, num_node:].shape)
            pca_params['Layer_{:d}/num_node'.format(i)] = num_node
        else:
            sgRDD = pointRDD.zip(knnRDD).zip(pcaRDD).flatMap(lambda x: pus.sg_cw(x[0][0], x[1], x[0][1]))
            sgRDD.persist()
            kernels, energy, num_node_next = pus.pca_cw(sgRDD, pre_energy, threshold)
            if i == len(n_sample) - 1:
                num_node_next = [0 for j in range(num_node)]
            bias = np.max(np.array(sgRDD.map(lambda x: LA.norm(x, axis=0)).collect()), axis=0)
            e = np.zeros((kernels.shape[0], kernels.shape[-1]))
            e[:, 0] = bias
            pcaRDD = sgRDD.map(lambda x: x + bias).map(
                lambda x: np.array([np.dot(x[:, j], kernels[j].T) for j in range(kernels.shape[0])])).map(lambda x: x - e)

            pca_fea = np.array(pcaRDD.collect())
            pca_fea = pca_fea.reshape((-1, n_newpoint[i], pca_fea.shape[1], pca_fea.shape[2]))
            pca_leaf_fea = np.concatenate([pca_fea[:, :, j, num_node_next[j]:] for j in range(num_node)], axis=-1)
            print('Hop ', i, ': ', pca_leaf_fea.shape)
            if i != len(n_sample) - 1:
                pca_nleaf_fea = np.concatenate([pca_fea[:, :, j, :num_node_next[j]] for j in range(num_node)], axis=-1)
                pcaRDD = sc.parallelize(pca_nleaf_fea, num_partition)

            pre_energy = np.concatenate([energy[j][:num_node_next[j]] for j in range(num_node)], axis=-1)
            num_node = np.sum(num_node_next)
            pca_params['Layer_{:d}/num_node'.format(i)] = num_node
            pca_params['Layer_{:d}/num_node_next'.format(i)] = num_node_next
            pca_params['Layer_{:d}/bias'.format(i)] = bias
        pca_params['Layer_{:d}/kernel'.format(i)] = kernels
        fea.append(pus.extract_single(pca_leaf_fea))
        pointRDD = fpsRDD
        sgRDD.unpersist()
        knnRDD.unpersist()
    fpsRDD.unpersist()
    pcaRDD.unpersist()
    pointRDD.unpersist()
    fea = np.concatenate(fea, axis=-1)
    return pca_params, fea


def pointhop_pred(data, n_newpoint, n_sample, pca_params, num_partition):
    point_data = data
    pcaRDD = None
    fea = []
    pointRDD = sc.parallelize(point_data, num_partition)
    for i in range(len(n_sample)):
        if len(point_data) == n_newpoint:
            fpsRDD = pointRDD
        else:
            fpsRDD = pointRDD.map(lambda x: pus.fps(x, n_newpoint[i]))
        fpsRDD.persist()

        knnRDD = fpsRDD.zip(pointRDD).map(lambda x: pus.knn(x[0], x[1], n_sample[i]))
        knnRDD.persist()
        kernels = pca_params['Layer_{:d}/kernel'.format(i)]

        if i == 0:
            num_node = pca_params['Layer_{:d}/num_node'.format(i)]
            sgRDD = pointRDD.zip(knnRDD).flatMap(lambda x: pus.sg(x[0], x[0], x[1]))
            sgRDD.persist()
            pcaRDD = sgRDD.map(lambda x: np.dot(x, kernels.T))
            pca_fea = np.array(pcaRDD.collect())
            pca_fea = pca_fea.reshape((-1, n_newpoint[i], pca_fea.shape[-1]))
            pcaRDD = sc.parallelize(pca_fea[:, :, :num_node], num_partition)
            pca_leaf_fea = pca_fea[:, :, num_node:]
            print('Hop ', i, ': ', pca_fea[:, :, num_node:].shape)
        else:
            num_node_next = pca_params['Layer_{:d}/num_node_next'.format(i)]
            sgRDD = pointRDD.zip(knnRDD).zip(pcaRDD).flatMap(lambda x: pus.sg_cw(x[0][0], x[1], x[0][1]))
            sgRDD.persist()
            bias = pca_params['Layer_{:d}/bias'.format(i)]
            e = np.zeros((kernels.shape[0], kernels.shape[-1]))
            e[:, 0] = bias
            pcaRDD = sgRDD.map(lambda x: x + bias).map(
                lambda x: np.array([np.dot(x[:, j], kernels[j].T) for j in range(kernels.shape[0])])).map(lambda x: x - e)

            pca_fea = np.array(pcaRDD.collect())
            pca_fea = pca_fea.reshape((-1, n_newpoint[i], pca_fea.shape[1], pca_fea.shape[2]))
            pca_leaf_fea = np.concatenate([pca_fea[:, :, j, num_node_next[j]:] for j in range(num_node)], axis=-1)
            print('Hop ', i, ': ', pca_leaf_fea.shape)
            if i != len(n_sample) - 1:
                pca_nleaf_fea = np.concatenate([pca_fea[:, :, j, :num_node_next[j]] for j in range(num_node)], axis=-1)
                pcaRDD = sc.parallelize(pca_nleaf_fea, num_partition)

            num_node = pca_params['Layer_{:d}/num_node'.format(i)]
        fea.append(pus.extract_single(pca_leaf_fea))
        pointRDD = fpsRDD
        sgRDD.unpersist()
        knnRDD.unpersist()
    fpsRDD.unpersist()
    pcaRDD.unpersist()
    pointRDD.unpersist()
    fea = np.concatenate(fea, axis=-1)
    return fea


if __name__ == '__main__':
    time_start = time.time()

    initial_point = 1024
    n_newpoint = [1024, 128, 128, 64]
    n_sample = [64, 64, 64, 64]
    threshold = 0.0001

    train_data, train_label = modelnet_data.data_load(initial_point, os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), True)
    test_data, test_label = modelnet_data.data_load(initial_point, os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), False)
    train_data = train_data
    train_label = train_label
    test_data = test_data
    test_label = test_label
    print('Train data loaded!')

    pca_params, feature_train = pointhop_train(train_data, n_newpoint, n_sample, threshold, num_partition=1000)
    print(feature_train.shape)

    feature_test = pointhop_pred(test_data, n_newpoint, n_sample, pca_params, num_partition=200)
    print(feature_test.shape)

    clf = pus.rf_classifier(feature_train, np.squeeze(train_label))
    pred_train = clf.predict(feature_train)
    acc_train = accuracy_score(train_label, pred_train)
    print('RF Classification train accuracy: ', acc_train)

    pred_test = clf.predict(feature_test)
    acc_test = accuracy_score(test_label, pred_test)
    print('RF Classification test accuracy: ', acc_test)

    weight = pus.llsr_train(feature_train, train_label, 40)
    prob_train, pred_train = pus.llsr_pred(feature_train, weight)
    acc_train = accuracy_score(train_label, pred_train)
    print('LLSR Classification train accuracy: ', acc_train)

    prob_test, pred_test = pus.llsr_pred(feature_test, weight)
    acc_test = accuracy_score(test_label, pred_test)
    print('LLSR Classification test accuracy: ', acc_test)

    weight = pus.llsr_train_weighted(feature_train, train_label, 40, epsilon=0.2)
    prob_train, pred_train = pus.llsr_pred(feature_train, weight)
    acc_train = accuracy_score(train_label, pred_train)
    print('WLLSR Classification train accuracy: ', acc_train)

    prob_test, pred_test = pus.llsr_pred(feature_test, weight)
    acc_test = accuracy_score(test_label, pred_test)
    print('WLLSR Classification test accuracy: ', acc_test)

    sc.stop()
    time_end = time.time()
    print('Duration:', (time_end - time_start) / 60.0, 'mins')



