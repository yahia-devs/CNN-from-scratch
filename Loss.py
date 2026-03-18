
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self._preds = None
        self._labels_oh = None

    def forward(self, predictions, labels):

        bs = predictions.shape[0]
        n_classes = predictions.shape[1]
        self._labels_oh = np.zeros((bs, n_classes))
        self._labels_oh[np.arange(bs), labels] = 1
        self._preds = np.clip(predictions, 1e-12, 1.0 - 1e-12)

        loss = -np.sum(self._labels_oh * np.log(self._preds)) / bs
        return loss

    def backward(self):
        return self._preds - self._labels_oh