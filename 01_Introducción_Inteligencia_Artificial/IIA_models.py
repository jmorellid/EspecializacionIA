import numpy as np

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model


class LinearRegression(BaseModel):

    def fit(self, X, y):
        W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return X.dot(self.model)

    def gradient_descent(self, X, y, alpha=0.01, epochs=100):
        n = X.shape[0]
        m = X.shape[1]

        W = np.random.randint(-0, 10, size=[m, 1])

        for i in range(epochs):
            prediction = np.matmul(X, W)

            error = y - prediction

            grad_sum = np.sum(error * X, axis=0)
            grad_mul = -2 / n * grad_sum
            gradient = grad_mul.T.reshape(-1, 1)

            W = W - (alpha * gradient)

        self.model = W

        return W

    def gradient_stochastic(self, X, y, alpha=0.01, epochs=100):
        n = X.shape[0]
        m = X.shape[1]

        W = np.random.randint(-100, 100, size=[m, 1])

        for i in range(epochs):

            idx = np.random.permutation(n)
            X = X[idx]
            y = y[idx]

            for j in range(n):
                prediction = np.matmul(X[j, :], W)

                error = y[j] - prediction

                grad_sum = error * X[j, :]
                grad_mul = -2 / n * grad_sum
                gradient = grad_mul.T.reshape(-1, 1)

                W = W - (alpha * gradient)

        self.model = W

        return W

    def mini_batch_gradient_descent(self, X, y, alpha=0.01, epochs=100, b=15):
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

                prediction = np.matmul(X_batch, W)

                error = y_batch - prediction

                grad_sum = np.sum(error * X_batch, axis=0)
                grad_mul = -2 / n * grad_sum
                gradient = grad_mul.T.reshape(-1, 1)

                W = W - (alpha * gradient)

        self.model = W

        return W


class Kmeans(BaseModel):

    def redefine_centroids(self, X, centroids, n_clusters):
        distance = np.sqrt(np.sum((centroids[: ,None] - X )**2, axis=2))
        centroid_with_min_distance = np.argmin(distance, axis=0)

        for i in range(centroids.shape[0]):
            centroids[i] = np.mean( X[centroid_with_min_distance == i, :], axis = 0)
        return centroids, centroid_with_min_distance

    def fit(self, X, n_clusters, MAX_ITER):
        centroids = np.eye(n_clusters, X.shape[1] ) *10 + np.random.random(size=[n_clusters, X.shape[1]] ) *2

        for i in range(MAX_ITER):
            centroids, clusters = self.redefine_centroids(X, centroids, n_clusters)

        self.model = centroids

        return  NotImplemented

    def predict(self, X):

        centroids = self.model
        distance = np.sqrt(np.sum((centroids[:, None] - X) ** 2, axis=2))
        centroid_with_min_distance = np.argmin(distance, axis=0)

        return centroid_with_min_distance


class LogisticRegression(BaseModel):
    def fit(self, X, y, alpha=0.01, epochs=100, b=15):
        n = X.shape[0]
        m = X.shape[1]

        W = np.random.randint(0, 10, size=[m, 1])

        W = np.random.randn(m).reshape(m, 1)

        for i in range(amt_epochs):
            idx = np.random.permutation(X.shape[0])
            X_train = X[idx]
            y_train = y[idx]

            batch_size = int(len(X_train) / b)
            for i in range(0, len(X_train), batch_size):
                end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
                batch_X = X_train[i: end]
                batch_y = y_train[i: end]

                exponent = np.sum(np.transpose(W) * batch_X, axis=1)
                prediction = 1 / (1 + np.exp(-exponent))
                error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)

                grad_sum = np.sum(error * batch_X, axis=0)
                grad_mul = 1 / b * grad_sum
                gradient = np.transpose(grad_mul).reshape(-1, 1)

                W = W - (lr * gradient)

        self.model = W

        return W
    # def fit(self, X, y, alpha=0.01, epochs=100, b=15):
    #     n = X.shape[0]
    #     m = X.shape[1]
    #
    #     W = np.random.randint(0, 10, size=[m, 1])
    #
    #     for i in range(epochs):
    #
    #         idx = np.random.permutation(n)
    #         X = X[idx]
    #         y = y[idx]
    #
    #         batch_size = int(n / b)
    #
    #         for j in range(0, n, batch_size):
    #             end_batch = j + batch_size if batch_size + j <= n else n
    #
    #             X_batch = X[j:end_batch, :]
    #             y_batch = y[j:end_batch, :]
    #
    #             expo = np.exp(- np.matmul(X_batch, W))
    #             sigmoid = 1 / (1 + expo)
    #             h = (sigmoid - y_batch)
    #             gradient = h * X_batch
    #
    #             grad_sum = np.sum(gradient, axis=0)
    #             grad_mul = -alpha / (batch_size) * grad_sum
    #
    #             W = W + grad_mul.reshape(-1, 1)
    #     self.model = W
    #     return W

    def fit_stocha(self, X, y, alpha=0.01, epochs=1000):
        n = X.shape[0]
        m = X.shape[1]

        W = np.ones(m)

        for i in range(epochs):

            idx = np.random.permutation(n)
            X = X[idx]
            y = y[idx]

            for j in range(0, n):
                expo = np.exp(-W.T * X[j])
                sigmoid = 1 / (1 + expo)
                h = (sigmoid - y[j])
                gradient = h * X[j]

                W = W - alpha * gradient
        self.model = W
        return W


    def predict(self, X, threshold=0.5):
        W = self.model
        prediction = 1 / (1 + np.exp(-np.matmul(X, W)))

        y_proba = prediction
        y_pred = (prediction >= threshold) * 1

        return y_pred, y_proba