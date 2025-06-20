from numba import njit
import math

import numpy as np

@njit(fastmath=True, cache=True)
def fwd(chrom:  np.float32[:],    # len 26 (chromosome: weights + biases)
        s:      np.float32[:] ):  # len 3  (sensor readings)

    s0, s1, s2 = s[0], s[1], s[2]
    weights = chrom[:20]         # first 20 = all weights
    biases = chrom[20:]         # last 6  = all biases

    # ── hidden layer (4 neurons) ─────────────────────────────
    h0 = math.tanh(s0*weights[0]  + s1*weights[1]  + s2*weights[2]  + biases[0])
    h1 = math.tanh(s0*weights[3]  + s1*weights[4]  + s2*weights[5]  + biases[1])
    h2 = math.tanh(s0*weights[6]  + s1*weights[7]  + s2*weights[8]  + biases[2])
    h3 = math.tanh(s0*weights[9]  + s1*weights[10] + s2*weights[11] + biases[3])

    # ── output layer (2 neurons) ─────────────────────────────
    vL = math.tanh(h0*weights[12] + h1*weights[13] +
                   h2*weights[14] + h3*weights[15] + biases[4])

    vR = math.tanh(h0*weights[16] + h1*weights[17] +
                   h2*weights[18] + h3*weights[19] + biases[5])

    return vL, vR
