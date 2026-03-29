import numpy as np
from Convolution import Convolution

image = np.random.randn(1, 6, 6)
conv = Convolution(2, 1, 3)

sortie = conv.forward(image)
print("Forme sortie:", sortie.shape)
print("Sortie:", sortie)