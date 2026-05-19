import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ln = len(z)
        res = [0] * ln
        for i in range(ln):
            sig_i = 1 / (1 + np.exp(-1*z[i]))
            res[i] = round(sig_i, 5)

        return res


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ln = len(z)
        res = [0] * ln
        for i in range(ln):
            relu_i = max(0.0, z[i])
            res[i] = round(relu_i, 5)

        return res
