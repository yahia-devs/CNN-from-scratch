
import numpy as np


class ReLU:

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, d_out):
        return d_out * self._mask

    def update(self, lr):
        pass 


class Softmax:

    def __init__(self):
        self._out = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self._out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._out

    def backward(self, d_out):
        return d_out 

    def update(self, lr):
        pass