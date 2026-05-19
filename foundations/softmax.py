import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        ln = len(z)             # length of the array
        mx = np.max(z)             # maximum logit in array, z
        res = [0] * ln
        # find denominator of softmax function
        sigma_z = np.sum(np.exp(z-mx))

        for i in range(ln):
            z_i = np.exp(z[i] - mx)/sigma_z
            res[i] = np.round(z_i, 4)

        return res

     
