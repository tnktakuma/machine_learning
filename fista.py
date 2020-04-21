from typing import List

import numpy as np
from scipy import linalg


class FISTA:
    """Fast Iterative Shrinkage Threshold Algorithm with linear objective.

    The Problem is
    $ min_beta (1 / 2n) || X beta - y ||_2^2 + lambda g(beta) $
    where g is
    $ g(x) = || x ||_1 $ or $ g(x) = || x ||_0 $ or $ g(x) = sum || x_c ||_2 $

    Args:
        coeff_reg: float
        regularizer: str
        eps: float
    """
    def __init__(self, coeff_reg=1e-3, regularizer='l1', eps=1e-8):
        self.coeff_reg = coeff_reg
        if regularizer == 'l1':
            self.proximal = self.proximal_l1
        elif regularizer == 'l0':
            self.proximal = self.proximal_l0
        elif regularizer == 'group':
            self.proximal = None
        else:
            raise ValueError
        self.beta = None
        self.eps = eps

    def proximal_l1(self, x: np.ndarray, eta: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - eta, 0)

    def proximal_l0(self, x: np.ndarray, eta: float) -> np.ndarray:
        z = x.copy()
        z[np.abs(x) < np.sqrt(2 * eta)] = 0
        return z

    def proximal_group(
            self,
            x: np.ndarray,
            eta: float,
            group: List[List[int]]) -> np.ndarray:
        z = x.copy()
        for g in group:
            norm = np.sqrt(np.sum(x[g] * x[g]))
            if norm <= eta:
                z[g] = 0
            else:
                z[g] *= 1 - eta / norm
        return z

    def fit(self, X, y, Lipchitz=None, group=None):
        _sample, _dim = X.shape
        A = X.T @ X / _sample
        b = y @ X / _sample
        if Lipchitz is None:
            Lipchitz = linalg.eigvalsh(A, eigvals=(_dim-1, _dim-1))[0]
        if group is not None and self.proximal is None:
            self.proximal = lambda x, eta: self.proximal_group(x, eta, group)
        eta = self.coeff_reg / Lipchitz
        self.beta = np.ones(_dim)
        pre_beta = np.zeros_like(self.beta)
        z = 0
        alpha = 1
        while np.linalg.norm(self.beta - pre_beta) > self.eps:
            pre_beta = self.beta
            pre_z = z
            pre_alpha = alpha
            gradient = A @ self.beta - b
            z = self.proximal(self.beta - gradient / Lipchitz, eta)
            alpha = 0.5 + np.sqrt(0.25 + alpha * alpha)
            self.beta = z + (pre_alpha - 1) / alpha * (z - pre_z)

    def transform(self, X):
        if self.beta is None:
            raise ValueError
        return X @ self.beta


if __name__ == '__main__':
    # Example
    X = np.random.randn(1000, 100)
    beta = np.random.randn(100) * 1e-4
    beta[:10] = np.random.randn(10)
    y = X @ beta
    fista = FISTA(regularizer='group')
    fista.fit(
        X,
        y,
        group=[
            [0, 1, 2],
            [3, 13, 23, 33],
            [4, 8, 12, 16],
            [5, 6, 7],
            [14, 15, 17, 18, 19, 20, 21, 22],
            [i for i in range(34, 100)]])
    print(fista.beta)
    print(beta)
