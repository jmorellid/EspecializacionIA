import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def sinthetic_dataset(SIZE):

    uniforme = np.random.random(SIZE)

    n1 = np.random.normal(5, 15, size=SIZE)
    n2 = np.random.normal(10, 2, size=SIZE)

    sint = (uniforme <= 0.25) * n1 + (uniforme > 0.25) * n2

    return sint

def expectation_maximization(X, n_clusters=2, epochs=10000):

    n = X.shape[0]
    m = X.shape[1]

    mean = np.random.random(size=[n_clusters, m])
    corr_matrix = np.random.random(size=[n_clusters, m, m])
    p_z = np.ones(n_clusters) / n_clusters
    pini = np.zeros([n_clusters, n])
    Ez = np.zeros([n_clusters, n])

    for j in range(epochs):
        for i in range(n_clusters):
            pini[i] = stats.multivariate_normal.pdf(X, mean[i], corr_matrix[i]) * p_z[i]

        sum_pini = np.sum(pini, axis=0)

        for i in range(n_clusters):
            Ez[i] = pini[i]/sum_pini

        Ez = np.nan_to_num(Ez)

        sum_ez = np.sum(Ez, axis=1)

        for i in range(n_clusters):
            mean[i] = np.sum(Ez[i].reshape(-1,1) * X ) / sum_ez[i]
            mean[i] = np.nan_to_num(mean[i])
            corr_matrix[i] = np.sum(Ez[i] * np.diag(np.matmul((X-mean[i]),(X-mean[i]).T))) / np.sum(Ez[i])

        p_z = Ez / n

    return mean


X = sinthetic_dataset(500).reshape(-1,1)

X_scaled = X / max(X)

w = expectation_maximization(X_scaled, 2)

