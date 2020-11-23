import numpy as np


def _logistic_loss(x):
    return np.log(1 + np.exp(-x))


def _logistic_loss_grad(x):
    return -1 / (1 + np.exp(x))


class SupervisedDictionaryLearning:
    """Supervised Dictionary Learning

    Parameters
    ----------
    n_components (int): number of atoms
    n_mu (int): number of mu
    max_iter (int): maximum of iterations
    tol (float): tolerance of error
    lambda0 (float): hyperparameter about reconstraction error
    lambda1 (float): hyperparameter about l1 regularization of sparse codes
    lambda2 (float): hyperparameter about l2 regularization of coefficients

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Dictionary atoms extracted from the data
    coef_ : array of shape (n_components,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Intercept added to the dicision function.
    n_iter_ : int
        Number of iterations run.
     """
    def __init__(self,
                 n_components,
                 n_mu=10,
                 max_iter=1000,
                 tol=1e-4,
                 lambda0=1e-3,
                 lambda1=1e-3,
                 lambda2=1e-3):
        self.n_components = n_components
        self.n_mu = n_mu
        self.max_iter = max_iter
        self.tol = tol
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _alpha_loss(self, alpha, X, y):
        f = alpha @ self.coef_ + self.intercept_
        error = X - alpha @ self.components_
        return _logistic_loss(y * f) + self.lambda0 * np.sum(
            error**2) + self.lambda1 * np.sum(np.abs(alpha))

    def _alpha_loss_grad(self, alpha, X, y):
        f = alpha @ self.coef_ + self.intercept_
        error = X - alpha @ self.components_
        return np.outer(
            y * _logistic_loss_grad(y * f),
            self.coef_) - 2 * self.lambda0 * error @ self.components_.T

    def _proximal(self, x, eta):
        return np.sign(x) * np.maximum(np.abs(x) - eta, 0)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        n_samples, n_features = X.shape
        self.components_ = np.random.randn(self.n_components, n_features)
        self.components_ /= np.linalg.norm(self.components_,
                                           axis=1,
                                           keepdims=True)
        self.coef_ = np.random.randn(self.n_components)
        self.intercept_ = np.random.randn()
        alpha_minus = np.zeros((n_samples, self.n_components))
        alpha_plus = np.zeros((n_samples, self.n_components))
        for self.mu in np.linspace(0, 1, self.n_mu):
            for self.n_iter_ in range(self.max_iter):
                alpha_minus = self.supervised_sparse_encode(alpha_minus, X, -y)
                alpha_plus = self.supervised_sparse_encode(alpha_plus, X, y)
                self.update_dict_and_coef(alpha_minus, alpha_plus, X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        pred : array, shape [n_samples]
            Predicted class label per sample.
        """
        alpha = np.zeros((len(X), self.n_components))
        alpha_minus = self.supervised_sparse_encode(alpha, X, -np.ones(len(X)))
        alpha_plus = self.supervised_sparse_encode(alpha, X, np.ones(len(X)))
        s_minus = self._alpha_loss(alpha_minus, X, -np.ones(len(X)))
        s_plus = self._alpha_loss(alpha_plus, X, np.ones(len(X)))
        pred = np.ones(len(X))
        pred[s_plus < s_minus] = -1
        return pred

    def supervised_sparse_encode(self, alpha, X, y):
        """Supervised sparse coding

        Parameters
        -----------
        alpha : array-like of shape (n_samples, n_components)
            Sparse code, where n_samples in the number of samples and
            n_components is the number of atoms.
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X

        Returns
        -------
        alpha : array-like of shape (n_samples, n_components)
        """
        eta = self.lambda1
        pre_alpha = self.tol
        z = 0
        a = 1
        while np.linalg.norm(alpha - pre_alpha) > self.tol:
            pre_alpha = alpha
            pre_z = z
            pre_a = a
            gradient = self._alpha_loss_grad(alpha, X, y)
            tmp = alpha - gradient
            z = np.sign(tmp) * np.maximum(np.abs(tmp) - eta, 0)
            a = 0.5 + np.sqrt(0.25 + a * a)
            alpha = z + (pre_a - 1) / a * (z - pre_z)
        return alpha

    def update_dict_and_coef(self, alpha_minus, alpha_plus, X, y):
        """Update Dictionary and Parameters

        Parameters
        ----------
        alpha_minus : array-like of shape (n_samples, n_components)
            Sparse code, where n_samples in the number of samples and
            n_components is the number of atoms.
        alpha_plus : array-like of shape (n_samples, n_components)
            Sparse code, where n_samples in the number of samples and
            n_components is the number of atoms.
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X
        """
        pre_components_ = np.zeros_like(self.components_)
        error_components_ = self.tol
        error_coef_ = self.tol
        error_intercept_ = self.tol
        while error_components_ + error_coef_ + error_intercept_ > self.tol:
            pre_components_ = self.components_
            diff = (self._alpha_loss(alpha_minus, X, -y) -
                    self._alpha_loss(alpha_plus, X, y))
            loss_diff_grad = _logistic_loss_grad(diff)
            minus_diff = X - alpha_minus @ self.components_
            plus_diff = X - alpha_plus @ self.components_
            diff_grad = (
                (loss_diff_grad[:, np.newaxis] * minus_diff).T @ alpha_minus -
                (loss_diff_grad[:, np.newaxis] * plus_diff).T @ alpha_plus)
            grad_components_ = -2 * self.lambda0 * (
                self.mu * diff_grad.T +
                (1 - self.mu) * alpha_plus.T @ plus_diff)
            plus_value = y * _logistic_loss_grad(
                y * (alpha_plus @ self.coef_ + self.intercept_))
            minus_value = -y * _logistic_loss_grad(
                -y * (alpha_minus @ self.coef_ + self.intercept_))
            diff_grad = (-(loss_diff_grad * minus_value) @ alpha_minus -
                         (loss_diff_grad * plus_value) @ alpha_plus)
            grad_coef_ = (
                self.mu * diff_grad.sum(axis=0) +
                (1 - self.mu) * alpha_plus.T @ (loss_diff_grad * plus_value))
            diff_grad = (np.sum((-(loss_diff_grad * minus_value).T -
                                 (loss_diff_grad * plus_value).T),
                                axis=0))
            grad_intercept_ = self.mu * diff_grad + (
                1 - self.mu) * loss_diff_grad @ y
            self.components_ -= grad_components_
            self.components_ /= np.linalg.norm(self.components_,
                                               axis=1,
                                               keepdims=True)
            self.coef_ -= grad_coef_
            self.intercept_ -= grad_intercept_
            error_components_ = np.linalg.norm(self.components_ -
                                               pre_components_)
            error_coef_ = np.linalg.norm(grad_coef_)
            error_intercept_ = np.linalg.norm(grad_coef_)


def main():
    """Sample code"""
    X = np.random.randn(1000, 100)
    y = 2 * (np.linalg.norm(X, axis=1) < 1) - 1
    sdl = SupervisedDictionaryLearning(10)
    sdl.fit(X, y)
    pred = sdl.predict(np.random.randn(10, 100))
    print(pred)


if __name__ == '__main__':
    main()
