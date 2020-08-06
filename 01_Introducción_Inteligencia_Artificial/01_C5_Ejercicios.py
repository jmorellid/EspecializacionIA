import numpy as np
import matplotlib.pyplot as plt
import os


def mini_batch_gradient_descent(X, y, alpha=0.01, epochs=100, b=15):
    n = X.shape[0]
    m = X.shape[1]

    W = np.random.randint(-100, 100, size=[m, 1])

    for i in range(epochs):

        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        batch_size = int(n / b)

        for j in range(0, n, batch_size):
            end_batch = j + batch_size if batch_size + j <= n else n

            X_batch = X[j:end_batch, :]
            y_batch = y[j:end_batch, :]

            gradient = np.matmul((1 / (1 + np.exp(-np.matmul(W.T, X_batch))) - y_batch), X_batch)

            grad_sum = np.sum(gradient, axis=0)
            grad_mul = -alpha / (batch_size) * grad_sum

            W = W - grad_mul

    return W


def log_reg_predict(X, W, threshold=0.5):
    prediction = 1 / (1 - np.exp(np.matmul(-W.T, X)))

    if prediction >= threshold:
        y_pred = 1
    else:
        y_pred = 0
    return y_pred


def split(Data, percentage):
    # divides the dataset by permutating, masking and slicing into train and split
    SIZE = Data.shape[0]
    data_total = Data
    idx = np.arange(0, SIZE)

    # permutate
    perm_idx = np.random.permutation(idx)

    # generate index slice
    train_idx = perm_idx[:int(SIZE * percentage)]
    test_idx = perm_idx[int(SIZE * percentage):]

    # slice dataset
    train_data = data_total[train_idx]
    test_data = data_total[test_idx]

    return train_data, test_data


os.chdir('C:/Users/jota_/00_Especialización_IA/')

FILE_PATH = '../EspecializacionIA/00_Datasets/01_Raw/clase_6_dataset.txt'

data = np.loadtxt(FILE_PATH, delimiter=',', dtype=str)

data[0][0] = data[0][0][3:]
data = data.astype('float64')

train, test = split(data, percentage=0.8)

X_train = train[:, :2]
y_train = train[:, 2].reshape(y_train.shape[0], 1)
X_test = test[:, :2]
y_test = test[:, 2].reshape(y_test.shape[0], 1)

y_train.shape

mini_batch_gradient_descent(X_train, y_train)
