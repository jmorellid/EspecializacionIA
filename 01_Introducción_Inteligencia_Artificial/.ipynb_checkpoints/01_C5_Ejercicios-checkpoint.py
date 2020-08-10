
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def mini_batch_gradient_descent(X, y, alpha=0.01, epochs=100, b=15):
    n = X.shape[0]
    m = X.shape[1]

    W = np.random.randint(0, 10, size=[m, 1])

    for i in range(epochs):

        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        batch_size = int(n / b)

        for j in range(0, n, batch_size):
            end_batch = j + batch_size if batch_size + j <= n else n

            X_batch = X[j:end_batch, :]
            y_batch = y[j:end_batch, :]

            expo = np.exp(- np.matmul(X_batch, W))
            sigmoid = 1 / (1 + expo)
            h = (sigmoid - y_batch)
            gradient = h * X_batch

            grad_sum = np.sum(gradient, axis=0)
            grad_mul = -alpha / (batch_size) * grad_sum

            W = W + grad_mul.reshape(-1,1)

    return W


def stochastic_gradient_descent(X, y, alpha=0.01, epochs=1000):
    n = X.shape[0]
    m = X.shape[1]

    W = np.ones(m)
    


    for i in range(epochs):

        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        for j in range(0, n):

            expo = np.exp(- np.matmul(X[j], W))
            sigmoid = 1 / (1 + expo)
            h = (sigmoid - y[j])
            gradient = h * X[j]

            W = W - alpha * gradient

    return W


def log_reg_predict(X, W, threshold=0.5):
    prediction = 1 / (1 + np.exp(-np.matmul(X, W)))

    y_proba = prediction
    y_pred = (prediction >= threshold)*1

    return y_pred, y_proba


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

FILE_PATH = 'C:/Users/jota_/00_Especialización_IA/EspecializacionIA/00_Datasets/01_Raw/clase_6_dataset.txt'

data = np.loadtxt(FILE_PATH, delimiter=',', dtype=str)

data[0][0] = data[0][0][3:]

data = data.astype('float64')

train, test = split(data, percentage=0.8)

X_train = train[:, :2]
y_train = train[:, 2].reshape(train.shape[0], 1)
X_test = test[:, :2]
y_test = test[:, 2].reshape(test.shape[0], 1)


X = np.vstack((X_train.T, np.ones(len(X_train)).reshape(1,len(X_train)))).T
y = y_test

w_stocha = stochastic_gradient_descent(X, y_train, alpha=0.001, epochs=100000)
w_mini = mini_batch_gradient_descent(X, y_train, alpha=0.001, epochs=100000)
print(w_stocha, w_mini)

X = np.vstack((X_test.T, np.ones(len(X_test)).reshape(1,len(X_test)))).T
y_pred, y_proba = log_reg_predict(X, w, threshold=0.6)

print(w, y_pred.T, y_test.T, np.sum(y_pred == y_test))

colors = np.apply_along_axis(lambda x: 'red' if x == 1 else 'blue',1 ,y_test)

a=30
b=60

w = w_mini[:,0]

sns.lineplot(x=[a,b], y=[-a*w[0]/w[1]-w[2]/w[1], -b*w[0]/w[1]-w[2]/w[1]])
sns.scatterplot(x=X_test[:,1], y=X_test[:,0], hue=colors)

plt.ylabel('y')
plt.xlabel('X')

plt.title(('Logistic Regression Model'))
plt.tight_layout()
plt.show()