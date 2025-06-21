from numba import njit
import math

import numpy as np

hidden_neurons = 6
output_neurons = 2
inputs = 3  # sensor readings
num_weights = hidden_neurons * inputs + output_neurons * hidden_neurons
num_biases = hidden_neurons + output_neurons

@njit(fastmath=True, cache=True)
def fwd(chrom:  np.ndarray,    # (chromosome: weights + biases)
        s:      np.ndarray ):  # len 3  (sensor readings)

    s0, s1, s2 = s[0], s[1], s[2]
    weights = chrom[:num_weights]         # first 20 = all weights
    biases = chrom[num_weights:]         # last 8  = all biases

    # ── hidden layer (6 neurons) ─────────────────────────────
    h0 = math.tanh(s0*weights[0]  + s1*weights[1]  + s2*weights[2]  + biases[0])
    h1 = math.tanh(s0*weights[3]  + s1*weights[4]  + s2*weights[5]  + biases[1])
    h2 = math.tanh(s0*weights[6]  + s1*weights[7]  + s2*weights[8]  + biases[2])
    h3 = math.tanh(s0*weights[9]  + s1*weights[10] + s2*weights[11] + biases[3])
    h4 = math.tanh(s0*weights[12]  + s1*weights[13]  + s2*weights[14]  + biases[4])
    h5 = math.tanh(s0*weights[15]  + s1*weights[16] + s2*weights[17] + biases[5])

    # ── output layer (2 neurons) ─────────────────────────────
    vL = math.tanh(h0*weights[18] + h1*weights[19] +
                   h2*weights[20] + h3*weights[21] + 
                   h4*weights[22] + h5*weights[23] + 
                   biases[6])
    
    vR = math.tanh(h0*weights[24] + h1*weights[25] +
                   h2*weights[26] + h3*weights[27] + 
                   h4*weights[28] + h5*weights[29] + 
                   biases[7])

    return vL, vR
