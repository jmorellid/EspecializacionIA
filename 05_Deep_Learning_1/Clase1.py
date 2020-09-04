import numpy as np
import scipy

X_raw = np.array([[0,0,1,1],[0,1,0,1]])

y_raw = np.array([0,1,1,0])


def sigmoid(t):
    return 1/(1 + np.exp(-t))


n_epoch = 10000
lr = 0.01

W111 = np.random.random(size=[1,1])
W121 = np.random.random(size=[1,1])
W211 = np.random.random(size=[1,1])
W221 = np.random.random(size=[1,1])

b111 = np.random.random(size=[1,1])
b211 = np.random.random(size=[1,1])

W112 = np.random.random(size=[1,1])
W222 = np.random.random(size=[1,1])
b222 = np.random.random(size=[1,1])

y_hat = np.zeros(shape=[4,1])
error = np.zeros(shape=[4,1])

for i in range(n_epoch):
    idx_shuffle = np.random.permutation(range(4))
    X = X_raw[:,idx_shuffle]
    y = y_raw[idx_shuffle]
    for j in range(4):
        z_1 = X[0,j] * W111 + X[1,j] * W121 + b111
        z_2 = X[0,j] * W211 + X[1,j] * W221 + b211

        a_1 = sigmoid(z_1)
        a_2 = sigmoid(z_2)
        y_hat[j] = a_1 * W112 + a_2 * W222 + b222

        error[j] = y[j] - y_hat[j]

        W112 = W112 + lr * 2 * error[j] * a_1
        W222 = W222 + lr * 2 * error[j] * a_2
        b222 = b222 + lr * 2 * error[j]

        W111 = W111 + lr * 2 * error[j] * W112 * sigmoid(z_1) * (1 - sigmoid(z_1)) * X[0,j]
        W121 = W121 + lr * 2 * error[j] * W112 * sigmoid(z_1) * (1 - sigmoid(z_1)) * X[1,j]
        W211 = W211 + lr * 2 * error[j] * W222 * sigmoid(z_2) * (1 - sigmoid(z_2)) * X[0,j]
        W221 = W221 + lr * 2 * error[j] * W222 * sigmoid(z_2) * (1 - sigmoid(z_2)) * X[1,j]
        b111 = b111 + lr * 2 * error[j] * W112 * sigmoid(z_1) * (1 - sigmoid(z_1))
        b211 = b222 + lr * 2 * error[j] * W222 * sigmoid(z_2) * (1 - sigmoid(z_2))

def predict(X):
    z_1 = X[0] * W111 + X[1] * W121 + b111
    z_2 = X[0] * W211 + X[1] * W221 + b211

    a_1 = sigmoid(z_1)
    a_2 = sigmoid(z_2)
    y_hat = a_1 * W112 + a_2 * W222 + b222

    return  y_hat
