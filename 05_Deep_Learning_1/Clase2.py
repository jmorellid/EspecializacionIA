import numpy as np
import scipy


class Layer():

    def __init__(self, nodes, n_inputs, batch, lr=0.01):
        self.nodes = nodes
        self.n_inputs = n_inputs
        self.W = np.random.random(size=[n_inputs, nodes])
        self.b = np.random.random(size=[nodes, 1])
        self.batch = batch
        self.lr = lr


    def forward(self):
        return NotImplemented


    def backward(self):
        return NotImplemented


    def update(self):
        return NotImplemented

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def g(self, t1):
        return self.sigmoid(t1) * (1 - self.sigmoid(t1))


class FullyConnected(Layer):

    def forward(self, X):
        W = self.W
        b = self.b

        Z = W.T @ X + b

        A = self.sigmoid(Z)

        self.Z = Z
        self.A = A


    def backward(self, dZ_f, A_b, W_f):
        dZ = W_f @ dZ_f * self.g(self.Z)

        grad_W = self.batch**-1 * dZ @ A_b.T
        grad_b = self.batch**-1 * np.sum(dZ, axis=1, keepdims=True)

        self.dZ = dZ
        self.grad_W = grad_W
        self.grad_b = grad_b

    def update(self):
        self.W = self.W - self.lr * self.grad_W.T
        self.b = self.b - self.lr * self.grad_b


class BinaryOutput(Layer):

    def forward(self, X):
        W = self.W
        b = self.b

        Z = W.T @ X + b

        Y_hat = self.sigmoid(Z)

        self.Z = Z
        self.Y_hat = Y_hat

    def backward(self, Y, A_b):
        dZ = -2 * (Y - self.Y_hat) * self.g(self.Z)

        grad_W = self.batch ** -1 * dZ @ A_b.T
        grad_b = self.batch ** -1 * np.sum(dZ, axis=1, keepdims=True)

        self.dZ = dZ
        self.grad_W = grad_W
        self.grad_b = grad_b

    def update(self):
        self.W = self.W - self.lr * self.grad_W.T
        self.b = self.b - self.lr * self.grad_b


X_raw = np.array([[0,0,1,1],[0,1,0,1]])
y_raw = np.array([0,1,1,0])

n = len(X)
b = 2
epochs = 100000

layer_1 = FullyConnected(nodes=3, n_inputs=2, batch=b, lr=0.1)
layer_2 = FullyConnected(nodes=2, n_inputs=3, batch=b, lr=0.1)
outputlayer = BinaryOutput(nodes=1, n_inputs=2, batch=b, lr=0.1)

n_net = [layer_1, layer_2, outputlayer]

for i in range(epochs):

    batch_size = int(n / b)
    idx_shuffle = np.random.permutation(range(len(X)))
    X = X_raw[:, idx_shuffle]
    y = y_raw[idx_shuffle]

    for j in range(0, n, batch_size):
        end_batch = j + batch_size if batch_size + j <= n else n

        for k, layer in enumerate(n_net):
            if k == 0:
                layer.forward(X[:,j:end_batch])
            else:
                layer.forward(n_net[k-1].A)

        for k, layer in enumerate(n_net[::-1]):
            if k == 0:
                layer.backward(y[j:end_batch], n_net[::-1][1].A )
            elif k != (len(n_net) - 1):
                layer.backward(dZ_f=n_net[::-1][k-1].dZ, A_b=n_net[::-1][k+1].A, W_f= n_net[::-1][k-1].W)
            else:
                layer.backward(dZ_f=n_net[::-1][k - 1].dZ, A_b= X[:,j:end_batch], W_f=n_net[::-1][k - 1].W)

        for layer in n_net:
            layer.update()

for k, layer in enumerate(n_net):
    if k == 0:
        layer.forward(X_raw)
    else:
        layer.forward(n_net[k-1].A)

print(outputlayer.Y_hat)