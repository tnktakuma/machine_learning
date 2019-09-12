import numpy as np


class LDA
    """Linear Discriminant Analysis

    args:
        x (np.ndarray): training data (samplesize X dimension)
        t (np.ndarray): label (samplesize)
    """
    def __init__(self, x: np.ndarray, t: np.ndarray):
        self.x = x
        self.t = t
        self.size, self.dim = x.shape

    def train(self):
        K = self.t.max() + 1
        mu = np.zeros((labels, self.dim))
        cov = np.zeros((self.dim, self.dim))
        prior = np.zeros(K)
        for k in range(K):
            mu[k] = np.mean(self.x[self.t == k], axis=0)
            dev = self.x[self.t == k] - self.mu[k]
            cov += dev.T @ dev
            prior[k] = np.log(np.sum(self.t == k))
        inv_cov = self.size * np.linalg.inv(cov)
        self.coeff = mu @ inv_cov
        self.intercept = prior - 0.5 * np.diag(mu @ inv_con @ mu.T)
    
    def predict(self, x: np.ndarray):
        y = x @ coeff.T + intercept
        return y.argmax(axis=1)
