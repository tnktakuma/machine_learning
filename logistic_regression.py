import numpy as np


class LogisticRegression:
    "Logistic Regression

    args:
        x (np.ndarray): training data (samplesize X dimension)
        t (np.ndarray): label (samplesize)
    """
    def __init__(self, x: np.ndarray, t: np.ndarray):
        self.x = x
        self.t = t
        self.K = t.max() + 1
        self.size, self.dim = x.shape

    def train(self, iteration=10, l2=1e-3):
        param_dim = (self.K - 1) * self.dim
        self.w = np.zeros(param_dim, np.float)
        t_mat = np.eye(K)[self.t][:, :-1]
        for i in range(iteration):
            y = np.exp(self.x @ self.w.reshape(self.K - 1, self.dim).T)
            y /= 1 + y.sum(axis=1, keepdims=True)
            grad = ((y - t_mat).T @ self.x).flatten()
            hesse = np.zeros((param_dim, param_dim), np.float)
            for ki in range(self.K - 1):
                dim_i = ki * self.dim
                hesse[dim_i : dim_i + self.dim, dim_i : dim_i + self.dim] = self.x.T @ np.diag(y[:, ki]) @ self.x
                for kj in range(self.K - 1):
                    dim_j = kj * self.dim
                    hesse[dim_i : dim_i + self.dim, dim_j : dim_j + self.dim] -= self.x.T @ np.diag(y[:, ki] * y[:, kj]) @ self.x
            self.w -= np.linalg.inv(hesse + l2 * np.identity(param_dim)) @ (grad + l2 * self.w)
    
    def predict(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        y = np.zeros((len(x), self.K), np.float)
        y[:, :-1] = x @ self.w.reshape(self.K - 1, self.dim).T
        return y.argmax(axis=1)
