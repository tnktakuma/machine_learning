import numpy as np


class PCA:
    """Principal Component Annalysis

    args:
        x (np.ndarray)
    """
    def __init__(self, x: np.ndarray):
        self.x = x
        self.size, self.dim = x.shape
        self.mean = x.sum(axis=0) / self.size
        self.dev = x - self.mean
        self.std = self.dev.T @ self.dev / self.size

    def run_first(self) -> np.ndarray:
        pre_vec = np.zeros(self.dim)
        vec = np.ones(self.dim)
        vec_diff = vec - pre_vec
        while vec_diff @ vec_diff > 0:
            pre_vec = vec
            vec = vec @ self.std
            vec /= np.sqrt(vec @ vec)
            vec_diff = vec - pre_vec
        return vec

    def run_all(self) -> np.ndarray:
        value, vector = np.linalg.eigh(self.std)
        return value[::-1], vector[:, ::-1].T

    def reconstruct(self, ccr=None, components=None) -> np.ndarray:
        value, vector = self.run_all()
        if ccr is not None:
            ccr_array = value.cumsum() / value.sum()
            components = np.sum(ccr_array < ccr) + 1
        if components is None:
            components = self.dim
        rex = self.dev @ vector[:components].T @ vector[:components]
        rex += self.mean
        return rex
