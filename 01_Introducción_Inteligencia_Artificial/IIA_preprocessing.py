import numpy as np

def sinthetic_data(mean, std, SIZE, n_clusters , separation):
    sint_data = np.zeros([SIZE, n_clusters])
    classes = np.zeros([SIZE])
    for i in range(n_clusters):
        from_ = i * int(SIZE /n_clusters)
        to_ = i * int(SIZE /n_clusters) + int(SIZE /n_clusters)
        sint_data[from_:to_ ,i] = separation[i] * np.array([np.random.normal(mean, std)])
        classes[from_:to_] = i + 1
    sint_data += np.random.normal(0, 3, size=sint_data.shape)

    return sint_data, classes

def replace_nans_colmean(data):
    nan_matrix = np.isnan(data)
    col_means = np.nanmean(data, axis=0)

    means_broadcasted = np.ones(shape=nan_matrix.shape) * col_means
    data[nan_matrix] = means_broadcasted[nan_matrix]

    return data

class Data(object):

    def __init__(self, path, structure, start_col_x=0, final_col_x=-1):
        self.dataset = self.build_dataset(path, structure, start_col_x, final_col_x)
        self.structure = structure
        self.start_col_x = start_col_x
        self.final_col_x = final_col_x

    def split(self, percentage):
        # divides the dataset by permutating, masking and slicing into train and split
        SIZE = self.dataset.shape[0]
        data_total = self.dataset
        idx = np.arange(0, SIZE)

        # permutate
        perm_idx = np.random.permutation(idx)

        # generate index slice
        train_idx = perm_idx[:int(SIZE * percentage)]
        test_idx = perm_idx[int(SIZE * percentage):]

        # slice dataset
        train_data = data_total[train_idx]
        test_data = data_total[test_idx]
        print(data_total.dtype.fields.keys())
        x_keys = [x for x in data_total.dtype.fields.keys()][:-1]
        y_key = [x for x in data_total.dtype.fields.keys()][-1:]

        X_train = train_data[x_keys]
        X_test = test_data[x_keys]
        y_train = train_data[y_key]
        y_test = test_data[y_key]

        return X_train, X_test, y_train, y_test

    def build_dataset(self, path, structure, start_col_x=0, final_col_x=-1):
        """
        Takes a numpy structured array, a filepath and the columns where the features start and finish.
        It takes a single target value column

        Input:
        Filepath- str
        structure- numpy structured array ej. Structure = np.dtype([('X', np.float32), ('y', np.float32)])
        start_col_x, final_col_x- int

        Output:
        Structured numpy array filled with values from file path.
        """
        # load numpy array from disk using a generator

        with open(path, encoding="utf-8-sig") as file:
            data_gen = [tuple(line.strip('\n').replace('"', '').split(',')) for line in file if not line.split(',')[1][2].isalpha()]
            data_total = np.fromiter(data_gen, structure)

        return data_total

def pca(X, d):
    # paso numero 1, centrar todas las variables
    X_centered = X - np.mean(X, axis=0)

    # Transponer la matriz y calcular la covarianza
    cov_x_t = np.cov(np.transpose(X))

    # calcular los autovalores y autovectores
    W, V = np.linalg.eig(cov_x_t)

    #
    W_index = np.argsort(W)[::-1]
    V_sorted = V[W_index]

    PCA = np.matmul(X_centered, V_sorted[:, :d])
    return PCA

def expand_ones_X(X):
    X_expanded = np.vstack((X.T, np.ones(len(X)).reshape(1, len(X)))).T
    return X_expanded


def adapt_data_order(X, order):
    """
    Toma un dataset X, devuelve el dataset que corresponde a un polinomio de orden 'orden'.
    Cada columna del dataset devuelto corresponde a X^i, con i creciente hasta orden.
    NO TIENE EN CUENTA EL ORDEN 0!

    In:
    ndarray

    Out:
    ndarray
    """
    X_repeat_order = np.repeat(X, order, axis=1)
    orders = np.array(range(1, order + 1))
    X_order = np.apply_along_axis(np.power, 0, X_repeat_order.T, orders).T

    return X_order

def adapt_multidimensional_data_order(X, order, structured=False):
    """
    Toma un dataset X multifeatures, devuelve el dataset que corresponde a un polinomio de orden 'orden'.
    Cada columna del dataset devuelto corresponde a X^i, con i creciente hasta orden.
    NO TIENE EN CUENTA EL ORDEN 0!

    In:
    ndarray

    Out:
    ndarray
    """

    lock = 0

    if not structured:
        m = X.shape[1]
        n = X.shape[0]

        X_new = np.empty(shape=[n, order])

        for i in range(m):
            if lock == 0:
                X_new = adapt_data_order(X[:,i].reshape(-1,1), order)
                lock += 1
            else:
                X_new = np.hstack([X_new, adapt_data_order(X[:,i].reshape(-1,1), order)])
    else:
        keys = [x for x in X.dtype.fields.keys()]
        for key in keys:
            if lock == 0:
                X_new = adapt_data_order(X[key].reshape(-1,1), order)
                lock += 1
            else:
                X_new = np.hstack([X_new, adapt_data_order(X[key].reshape(-1,1), order)])
    return X_new


def train_test_split(X, y, percentage):
        # divides the dataset by permutating, masking and slicing into train and split
        SIZE = X.shape[0]
        idx = np.arange(0, SIZE)

        # permutate
        perm_idx = np.random.permutation(idx)

        # generate index slice
        train_idx = perm_idx[:int(SIZE * percentage)]
        test_idx = perm_idx[int(SIZE * percentage):]

        # slice dataset

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].reshape(-1,1)
        y_test = y[test_idx].reshape(-1,1)

        return X_train, X_test, y_train, y_test

import sys
import numpy as np
sys.path.insert(1, '00_Especialización_IA/EspecializacionIA/01_Introducción_Inteligencia_Artificial')
from IIA_preprocessing import Data
X = Data
filepath = 'C:/Users/jota_/00_Especialización_IA/00_Recursos/01_DataSets/income.data.csv'
Structure = np.dtype([('index', np.float32), ('X', np.float32), ('y', np.float32)])
data = X(filepath, Structure)