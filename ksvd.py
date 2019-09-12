import numpy as np


class KSVD:
    def __init__(self, data: np.ndarray, m=300, k0=10):
        self.data = data
        self.m = m
        self.k0 = k0
        A = data[np.random.permutation(N)[:m]]
        self.A = A / np.linalg.norm(A, axis=1, keepdims=True)

    def train(self, eps=1e-3):
        err = self.data
        prerr = 0
        N, dim = self.data.shape
        while np.abs(np.sum(err ** 2) - np.sum(prerr ** 2)) > eps:
            # Calculate l0 norm problem by OMP
            X = np.zeros((N, self.m))
            for i in range(N):
                r = self.data[i]
                support = []
                for k in range(self.k0):
                    support.append(np.argmax(np.abs(r @ self.A.T)))
                    As = self.A[support]
                    x = self.data[i] @ As.T @ np.linalg.inv(As @ As.T)
                    r = self.data[i] - x @ As
                for k, j in enumerate(support):
                    X[i, j] = x[k]
            # Learn dictionary
            for j in range(self.m):
                index = []
                for i in range(N):
                    if X[i, j] != 0:
                        index.append(j)
                        X[i, j] = 0
                if index == []:
                    self.A[i] = self.data[np.argmax(np.sum(err ** 2, axis=0))]
                    self.A[i] = self.A[i] / np.linalg.norm(self.A[i])
                else:
                    for pre in range(i-1):
                        if np.abs(self.A[i] @ self.A[pre]) > 0.99:
                            self.A[i] = self.data[np.argmax(np.sum(err ** 2, axis=0))]
                            self.A[i] = self.A[i] / np.linalg.norm(self.A[i])
                            break
                    E = self.data[index] - self.A.dot(X[index])
                    U, s, V = np.linalg.svd(E, full_matrices=False)
                    self.A[i] = U[:, 0]
                    X[i, index] = s[0] * V[0, :]
            prerr = err
            err = self.data - self.A.dot(X)
