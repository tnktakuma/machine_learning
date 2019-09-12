import numpy as np
from scipy.special import expit


class Layer:
    def __init__(self, units_num: int, activate: str):
        self.outputs = None
        self.diff = None
        self.errors = None
        self.units_num = units_num
        self.activate = activate
        self.next_layer = None
        self.before_layer = None
        self.weights = np.random.randn(self.before_layer.units_num, units_num)
        
    def calculations(self, inputs):
        if self.activate == 'sigmoid':
            self.outputs = expit(inputs)
            self.diff = (1. - expit(inputs)) * expit(inputs)
        elif self.activate == 'tanh':
            self.outputs = np.tanh(inputs)
            self.diff = np.cosh(inputs) ** (-2)
        elif self.activate == 'relu':
            self.outputs = np.maximux(inputs, 0.)
            self.diff = (inputs > 0).astype(np.float)
        elif self.activate == 'identity':
            self.outputs = inputs
            self.diff = np.ones_like(inputs)

    def feed_forward(self):
        inputs = self.before_layer.outputs @ self.weights
        self.calculations(inputs)
        if self.next_layer is not None:
            self.next_layer.feed_forward()
    
    def back_propagate(self, errors: np.ndarray):
        self.errors = errors
        if self.before_layer is not None:
            prop = self.before_layer.diff * (self.weights @ errors)
            self.before_layer.back_propagate(prop)
        
    def update_weights(self, lr=1e-3):
        self.weights = self.weights - lr * np.outer(self.diff, self.before_layer.outputs)
