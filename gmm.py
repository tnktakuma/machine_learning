import numpy as np
from scipy.special import logsumexp


class GMM:
    """Gaussian Mixture Model

    args:
        X (np.ndarray): Data (samplesize X dimension)
        K (int): Number of Mixtures
    """
    def __init__(self, X: np.ndarray, K: int):
        self.X = X
        self.K = K
        self.N, self.D = X.shape
        self.pi = np.ones(K) / K
        self.mu = np.random.randn(K, D)
        self.Gamma = np.repeat(np.eye(D)[np.newaxis, :, :], K, axis=0)

    def log_gaussian(self, n: int, k: int) -> np.float:
        x = self.X[n]
        mu = self.mu[k]
        Gamma = self.Gamma[k]
        coeff = 0.5 * (np.log(np.linalg.det(Gamma))
        coeff -= self.D * np.log(2. * np.pi))
        return coeff - 0.5 * (x - mu) @ Gamma @ (x - mu)
    
    def log_likelihood(self):
        return np.sum([logsumexp([np.log(self.pi[k]) + self.log_gaussian(n, k)
            for k in range(self.K)]) for n in range(self.N)])
    
    def train(self):
        # Iterate E-step and M-step alternately
        loglkh = self.log_likelihood()
        pre_loglkh = -np.inf
        while loglkh - pre_loglkh > 0:
            # E-step
            self.gamma = np.array([[self.pi[k] * np.exp(self.log_gaussian(n, k))
                for k in range(self.K)] for n in range(self.N)])
            self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)
            # M-step
            self.pi = np.sum(self.gamma, axis=0) / self.N
            self.mu = self.gamma.T @ self.X / self.pi[:, np.newaxis] / self.N
            for k in range(self.K):
                dev = self.X - self.mu[k]
                Gamma[k] = np.linalg.inv(dev.T @ np.diag(self.gamma[:, k]) @ dev / self.pi[k] / self.N)
            # Calculate log likelyhood
            pre_loglkh = loglkh
            loglkh = self.log_likelihood()
