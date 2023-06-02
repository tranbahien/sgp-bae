import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd

from scipy import stats


def load_jura(data_dir):
    name_corr = {'Xloc': 'x',
                 'Yloc': 'y',
                 'Landuse': 'land',
                 'Rock': 'rock'}
    # Load training data.
    train = pd.read_csv(os.path.join(data_dir, 'prediction.dat'))
    columns = [name_corr[c] if c in name_corr else c for c in train.columns]
    train.columns = columns
    train.set_index(['x', 'y'], inplace=True)

    test = pd.read_csv(os.path.join(data_dir, 'validation.dat'))
    test.columns = [name_corr[c] if c in name_corr else c for c in test.columns]
    test.set_index(['x', 'y'], inplace=True)

    # Setup according to experiment.
    train = pd.concat([train[['Ni', 'Zn', 'Cd']], test[['Ni', 'Zn']]])
    test = test[['Cd']]

    return train, test


def plot_mnist(arr, recon_arr, title, nr_images=8, seed=0):
    """

    :param arr:
    :param recon_arr:
    :param title:
    :param nr_images:
    :param seed:
    :return:
    """
    random.seed(seed)
    assert nr_images % 8 == 0

    indices = random.sample(list(range(len(arr))), nr_images)
    plt.figure(figsize=(10, 10*int(nr_images/8)))
    plt.suptitle(title)
    for i in range(int(nr_images*2)):
        plt.subplot(int(nr_images / 2), 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % 2 == 0:
            plt.imshow(arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Ground truth, id: {}".format(indices[i // 2]))
        else:
            plt.imshow(recon_arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Recon image, id: {}".format(indices[i // 2]))
    # plt.tight_layout()
    plt.draw()


def generate_init_inducing_points(train_data_path, n=5, nr_angles=16, seed_init=0, remove_test_angle=None,
                                  PCA=False, M=8, seed=0):
    """
    Generate initial inducing points for rotated MNIST data.
    For each angle we sample n object vectors from empirical distribution of PCA embeddings of training data.

    :param n: how many object vectors per each angle to sample
    :param nr_angles: number of angles between [0, 2pi)
    :param remove_test_angle: if None, test angle is kept in inducing point set. Else if index between 0 and nr_angles-1
        is passed, this angle is removed (to investigate possible data leakage through inducing points).
    :param PCA: whether or not to use PCA initialization
    :param M: dimension of GPLVM vectors
    """

    random.seed(seed)

    data = pickle.load(open(train_data_path, "rb"))
    data = data['aux_data']

    angles = np.linspace(0, 2 * np.pi, nr_angles + 1)[:-1]
    inducing_points = []

    if n < 1:
        indices = random.sample(list(range(nr_angles)), int(n*nr_angles))
        n = 1
    else:
        indices = range(nr_angles)

    for i in indices:

        # skip test angle
        if i == remove_test_angle:
            continue

        # for reproducibility
        seed = seed_init + i

        if PCA:
            obj_vectors = []
            for pca_ax in range(2, 2 + M):
                # sample from empirical dist of PCA embeddings
                obj_vectors.append(stats.gaussian_kde(data[:, pca_ax]).resample(int(n), seed=seed))

            obj_vectors = np.concatenate(tuple(obj_vectors)).T
        else:
            obj_vectors = np.random.normal(0, 1.5, int(n)*M).reshape(int(n), M)

        obj_vectors = np.hstack((np.full((int(n), 1), angles[i]), obj_vectors))  # add angle to each inducing point
        inducing_points.append(obj_vectors)

    inducing_points = np.concatenate(tuple(inducing_points))
    id_col = np.array([list(range(len(inducing_points)))]).T
    inducing_points = np.hstack((id_col, inducing_points))
    return inducing_points



def import_rotated_mnist(MNIST_path, ending, batch_size, digits="3", N_t=None, global_index=False):
    """

    Support for loading of data and batching via tf.data.Dataset API.

    :param MNIST_path:
    :param ending:
    :param batch_size:
    :param N_t: How many angels in train set for each image in test set
                (since reGPVAE implementation is based on not_shuffled data).
    :param global_index: if True, add global index to the auxiliary data (used in SVIGP_Hensman)

    :return:
    """

    # TODO: here we load entire data in the memory. For MNIST that is fine, for larger datasets will have to
    #  implement it in more efficient way

    # train data
    train_data_dict = pickle.load(open(MNIST_path + 'train_data' + ending, 'rb'))
    if N_t is not None:
        flatten = lambda l: [item for sublist in l for item in sublist]
        digit_mask = [True] * N_t + [False] * (15 - N_t)

        mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_data_dict['aux_data'])/15))]
        mask = flatten(mask)
        train_data_dict['images'] = train_data_dict['images'][mask]
        train_data_dict['aux_data'] = train_data_dict['aux_data'][mask]

        # add train images without test angles
        if N_t < 15:
            train_not_in_test_data_dict = pickle.load(open(MNIST_path + 'train_not_in_test_data' + ending, 'rb'))

            n = int(len(digits) * 270 * (15 - N_t) / N_t) * N_t

            mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_not_in_test_data_dict['aux_data']) / 15))]
            mask = flatten(mask)

            train_data_dict['images'] = np.concatenate((train_data_dict['images'],
                                                        train_not_in_test_data_dict['images'][mask][:n, ]), axis=0)
            train_data_dict['aux_data'] = np.concatenate((train_data_dict['aux_data'],
                                                          train_not_in_test_data_dict['aux_data'][mask][:n, ]), axis=0)

    if global_index:
        add_global_index = lambda arr: np.c_[list(range(len(arr))), arr]
        train_data_dict['aux_data'] = add_global_index(train_data_dict['aux_data'])


    # eval data
    eval_data_dict = pickle.load(open(MNIST_path + 'eval_data' + ending, 'rb'))
    if global_index:
        eval_data_dict['aux_data'] = add_global_index(eval_data_dict['aux_data'])

    # test data
    test_data_dict = pickle.load(open(MNIST_path + 'test_data' + ending, 'rb'))
    if global_index:
        test_data_dict['aux_data'] = add_global_index(test_data_dict['aux_data'])

    return train_data_dict, eval_data_dict, test_data_dict
