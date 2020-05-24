import os
import time
import numpy as np
import modelnet_data
from sklearn import ensemble
from numpy.linalg import eigh
from pyspark import SparkConf
from numpy import linalg as LA
from pyspark import SparkContext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fps_knn_sg(sample, fea, n_newpoint, n_sample):
    '''
    :param sample:(N, 3)
    :param fea:(N, dim)
    :param n_newpoint: K
    :param n_sample: M
    :return:(K, 8, dim)
    '''
    if len(sample) == n_newpoint:
        fps_sample = sample
    else:
        fps_sample = fps(sample, n_newpoint)
    nn_idx = knn(fps_sample, sample, n_sample)
    sg_fea = sg(sample, fea, nn_idx)
    return sg_fea


def fps(sample, n_newpoint):
    '''
    :param sample:(N, 3)
    :param n_newpoint: K
    :return:(K, 3)
    '''
    fps_sample = []
    farthest = np.random.randint(len(sample))
    distance = np.ones((len(sample),), dtype=int) * 1e10
    for k in range(n_newpoint):
        fps_sample.append(sample[farthest])
        dist = np.sum((sample - sample[farthest, :]) ** 2, axis=-1)
        idx = dist < distance
        distance[idx] = dist[idx]
        farthest = np.argmax(distance, axis=-1)
    return np.array(fps_sample)


def calc_distances(new_pts, pts):
    '''
    :param new_pts:(K, 3)
    :param pts:(N, 3)
    :return:(N, K)
    '''
    tmp_trans = np.transpose(np.array(new_pts), [1, 0])
    pts = np.array(pts)
    xy = np.matmul(pts, tmp_trans)
    pts_square = (pts**2).sum(axis=1, keepdims=True)
    tmp_square_trans = (tmp_trans**2).sum(axis=0, keepdims=True)
    return np.squeeze(pts_square + tmp_square_trans - 2 * xy)


def knn(new_pts, pts, n_sample):
    '''
    :param new_pts:(K, 3)
    :param pts:(N, 3)
    :param n_sample:int
    :return: nn_idx (K, n_sample)
    '''
    distance_matrix = calc_distances(new_pts, pts)
    nn_idx = np.argpartition(distance_matrix, (0, n_sample), axis=0)[:n_sample, :]
    nn_idx = np.transpose(nn_idx, [1, 0])
    return nn_idx


def sg(sample, fea, nn_idx):
    '''
    :param sample:(N, 3)
    :param fea:(N, n_sample, dim)
    :return: nn_idx (K, 8, dim)
    '''
    pts_fea = np.concatenate([sample, fea], axis=-1)
    nn_fea = []
    for i in range(nn_idx.shape[0]):
        nn_fea.append(pts_fea[nn_idx[i], :])
    nn_fea = np.array(nn_fea)
    pc_n = nn_fea[..., :3]
    pc_fea = nn_fea[..., 3:]
    pc = np.expand_dims(pc_n[:, 0, :], axis=1)
    pc_c = pc_n - pc
    pc_idx = []
    pc_idx.append(pc_c[:, :, 0] >= 0)
    pc_idx.append(pc_c[:, :, 0] <= 0)
    pc_idx.append(pc_c[:, :, 1] >= 0)
    pc_idx.append(pc_c[:, :, 1] <= 0)
    pc_idx.append(pc_c[:, :, 2] >= 0)
    pc_idx.append(pc_c[:, :, 2] <= 0)

    pc_bin = []
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=2))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=2))

    value = np.multiply(pc_fea, pc_bin)
    value = np.sum(value, axis=2, keepdims=True)
    num = np.sum(pc_bin, axis=2, keepdims=True)
    sg_fea = np.squeeze(value/num, axis=(2,))
    sg_fea = np.transpose(sg_fea, [1, 0, 2])
    sg_fea = sg_fea.reshape((sg_fea.shape[0], -1))
    return sg_fea


def sg_cw(sample, fea, nn_idx):
    '''
    :param sample:(N, 3)
    :param fea:(N, n_sample, dim)
    :return: nn_idx (K, 8, dim)
    '''
    sg_fea = []
    for i in range(fea.shape[-1]):
        fea_cw = fea[:, i].reshape((fea.shape[0], 1))
        pts_fea = np.concatenate([sample, fea_cw], axis=-1)
        nn_fea = []
        for i in range(nn_idx.shape[0]):
            nn_fea.append(pts_fea[nn_idx[i], :])
        nn_fea = np.array(nn_fea)
        pc_n = nn_fea[..., :3]
        pc_fea = nn_fea[..., 3:]
        pc = np.expand_dims(pc_n[:, 0, :], axis=1)
        pc_c = pc_n - pc
        pc_idx = []
        pc_idx.append(pc_c[:, :, 0] >= 0)
        pc_idx.append(pc_c[:, :, 0] <= 0)
        pc_idx.append(pc_c[:, :, 1] >= 0)
        pc_idx.append(pc_c[:, :, 1] <= 0)
        pc_idx.append(pc_c[:, :, 2] >= 0)
        pc_idx.append(pc_c[:, :, 2] <= 0)

        pc_bin = []
        pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=2))
        pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=2))

        value = np.multiply(pc_fea, pc_bin)
        value = np.sum(value, axis=2, keepdims=True)
        num = np.sum(pc_bin, axis=2, keepdims=True)
        sg_fea_cw = np.squeeze(value/num, axis=(2,))
        sg_fea_cw = np.transpose(sg_fea_cw, [1, 0, 2])
        sg_fea_cw = sg_fea_cw.reshape((sg_fea_cw.shape[0], -1))
        sg_fea.append(sg_fea_cw)
    sg_fea = np.transpose(sg_fea, [1, 2, 0])
    return sg_fea


def pca_cw(sgRDD, pre_energy, threshold):
    '''
    :param sg_fea:(M*K, dim, channel)
    :return: kernels (channel, dim, dim)
    :return: energy (channel, dim)
    '''
    kernels = []
    energies = []
    num_node_next = []
    dc = np.array(sgRDD.map(lambda x: np.mean(x, axis=0)).collect())
    sgRDD = sgRDD.map(lambda x: x - np.mean(x, axis=0))
    fe = np.squeeze(sgRDD.map(lambda x: (1, (x, 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
                    .map(lambda x: x[1][0] / float(x[1][1])).collect())
    sgRDD = sgRDD.map(lambda x: x - fe)

    num_channels = fe.shape[0]
    largest_eva = np.var(dc, axis=0) * num_channels
    dc_kernel = [1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_eva[i]) for i in range(len(largest_eva))]

    cov = sgRDD.map(lambda x: np.array([np.outer(x[:, i], x[:, i]) for i in range(x.shape[-1])])).sum() / dc.shape[0]
    col = cov.shape[-1]
    for i in range(cov.shape[0]):
        eva, eve = eigh(cov[i])
        inds = np.argsort(eva)
        kernel = eve.T[inds[-1:-(col + 1):-1]]
        eva = eva[inds[-1:-(col + 1):-1]]
        kernel = np.concatenate((dc_kernel[i], kernel), axis=0)[:num_channels]
        eva = np.concatenate(([largest_eva[i]], eva), axis=0)[:num_channels]
        energy = np.array([i / sum(eva) for i in eva]) * pre_energy[i]
        num_node_next += [np.sum(energy > threshold)]
        kernels.append(kernel)
        energies.append(energy)
    kernels = np.array(kernels)
    energies = np.array(energies)
    return kernels, energies, num_node_next


def pca(sgRDD):
    '''
    :param sg_fea:(M*K, 24)
    :return: kernels (24, 24)
    :return: energy (24)
    '''
    dc = np.array(sgRDD.map(lambda x: np.mean(x)).collect())
    sgRDD = sgRDD.map(lambda x: x - np.mean(x))
    fe = np.squeeze(sgRDD.map(lambda x: (1, (x, 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
                  .map(lambda x: x[1][0]/float(x[1][1])).collect())
    sgRDD = sgRDD.map(lambda x: x - fe)

    num_channels = fe.shape[0]
    largest_eva = [np.var(dc) * num_channels]
    dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_eva)

    cov = sgRDD.map(lambda x: np.outer(x, x)).sum()/dc.shape[0]
    col = cov.shape[1]
    eva, eve = eigh(cov)
    inds = np.argsort(eva)
    kernels = eve.T[inds[-1:-(col + 1):-1]]
    eva = eva[inds[-1:-(col + 1):-1]]
    kernels = np.concatenate((dc_kernel, kernels), axis=0)[:num_channels]
    eva = np.concatenate((largest_eva, eva), axis=0)[:num_channels]
    energy = np.array([i / sum(eva) for i in eva])
    return kernels, energy


def extract(feat):
    '''
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_samples, feature_dimension]
    # :param pooling: pooling method to be used
    :return: feature
    '''
    mean = []
    maxi = []
    l1 = []
    l2 = []

    for i in range(len(feat)):
        mean.append(feat[i].mean(axis=1, keepdims=False))
        maxi.append(feat[i].max(axis=1, keepdims=False))
        l1.append(np.linalg.norm(feat[i], ord=1, axis=1, keepdims=False))
        l2.append(np.linalg.norm(feat[i], ord=2, axis=1, keepdims=False))
    mean = np.concatenate(mean, axis=-1)
    maxi = np.concatenate(maxi, axis=-1)
    l1 = np.concatenate(l1, axis=-1)
    l2 = np.concatenate(l2, axis=-1)

    return [mean, maxi, l1, l2]


def extract_single(feat):
    '''
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_samples, feature_dimension]
    # :param pooling: pooling method to be used
    :return: feature
    '''
    feature = []
    feature.append(feat.mean(axis=1, keepdims=False))
    feature.append(feat.max(axis=1, keepdims=False))
    feature.append(np.linalg.norm(feat, ord=1, axis=1, keepdims=False))
    feature.append(np.linalg.norm(feat, ord=2, axis=1, keepdims=False))
    feature = np.concatenate(feature, axis=-1)
    return feature


def average_acc(label, pred_label):
    classes = np.arange(40)
    acc = np.zeros(len(classes))
    for i in range(len(classes)):
        ind = np.where(label == classes[i])[0]
        pred_test_special = pred_label[ind]
        acc[i] = len(np.where(pred_test_special == classes[i])[0]) / float(len(ind))
    return acc


def onehot_encoding(n_class, labels):
    targets = labels.reshape(-1)
    one_hot_targets = np.eye(n_class)[targets]
    return one_hot_targets


def llsr_train(feature, label, num_class):
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    if num_class is not None:
        y = onehot_encoding(num_class, label)
    else:
        y = label
    weight = np.matmul(LA.pinv(feature), y)
    return weight


def llsr_train_weighted(feature, label, num_class, epsilon):
    w = np.zeros((label.shape[0], label.shape[0]))
    f = []
    for i in range(num_class):
        idx = np.where(label == i)[0]
        f.append(1/(float(len(idx))/label.shape[0] + epsilon))
    for i in range(feature.shape[0]):
        w[i, i] = f[label[i][0]]

    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    if num_class is not None:
        y = onehot_encoding(num_class, label)
    else:
        y = label
    weight = np.matmul(LA.pinv(np.matmul(w, feature)), np.matmul(w, y))
    return weight


def llsr_pred(feature, weight):
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    feature = np.matmul(feature, weight)
    pred = np.argmax(feature, axis=1)
    return feature, pred


def rf_classifier(feat, y):
    '''
    Train random forest based on the provided feature.
    :param feat: [num_samples, feature_dimension]
    :param y: label provided
    :return: classifer
    '''
    clf = ensemble.RandomForestClassifier(n_estimators=128, bootstrap=False,
                                          n_jobs=-1)
    clf.fit(feat, y)
    return clf


if __name__ == '__main__':
    time_start = time.time()
    config = SparkConf().setAll(
        [('spark.driver.memory', '4g'),
         ('spark.executor.memory', '4g'),
         ('spark.driver.maxResultSize', '2g')]).setAppName('PCSEG').setMaster('local[*]')
    sc = SparkContext(conf=config)
    sc.setLogLevel("ERROR")

    train_data, train_label = modelnet_data.data_load(1024, os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), True)
    test_data, test_label = modelnet_data.data_load(1024, os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), False)
    train_data = train_data[:200]
    train_label = train_label[:200]
    test_data = test_data[:200]
    test_label = test_label[:200]
    print('Train data loaded!')

    pointRDD = sc.parallelize(train_data, 5)
    fpsRDD = pointRDD.map(lambda x: fps(x, 128))
    knnRDD = fpsRDD.zip(pointRDD).map(lambda x: knn(x[0], x[1], 64))
    sgRDD = pointRDD.zip(knnRDD).flatMap(lambda x: sg(x[0], x[0], x[1]))
    kernels, energy = pca(sgRDD)
    print(energy)
    # num_partition = 1500
    # train, stat = shapenet_data.data_load(1024, 'train')
    # train_data = train['data']
    # print('data loaded!')
    # sg_fea = np.array(sc.parallelize(train_data, num_partition).map(lambda x: fps_knn_sg(x, x, 20, 10)).collect())
    # s1, s2, s3 = sg_fea.shape
    # sg_fea = sg_fea.reshape((-1, s3))
    #
    # kernels, energy = pca(sc, num_partition, sg_fea)
    # pca_fea = np.array(sc.parallelize(sg_fea, num_partition).map(lambda x: np.dot(x, kernels[:5].T)).collect())
    # pca_fea = pca_fea.reshape((s1, s2, -1))
    # print(pca_fea.shape)

    sc.stop()
    time_end = time.time()
    print('Duration:', time_end - time_start)

