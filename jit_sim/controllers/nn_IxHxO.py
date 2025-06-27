from __future__ import annotations
import math, numpy as np
from numba import njit

class FeedForwardNNController:
    def __init__(self,
                 I:     np.int32    = 3,
                 H:     np.int32    = 4,
                 O:     np.int32    = 1):
        
        self.I       = I
        self.H       = H
        self.O       = O

    def chrom_len(self) -> int:
        """
        Returns the length of the chromosome for a feed-forward NN with
        I inputs, H hidden neurons, and O outputs.
        """
        return self.I * self.H + self.H + self.H * self.O + self.O
    
    def chrom_fmt(self) -> np.array:
        """
        Returns a 1-D array with the format of the chromosome:
        [H, O] where H is the number of hidden neurons,
        and O is the number of outputs.
        """
        return np.array([self.H, self.O], dtype=np.int32)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def fwd(chrom:      np.array,  # (chromosome: weights + biases)
            chrom_fmt:  np.array,
            I:          np.array):
        """
        chrom layout:
        W_in   (N_in * H)
        b_h    (H)
        W_out  (H * O)
        b_out  (O)
        Returns 1-D float32 array length O.
        """
        N_in = len(I)
        idx  = 0
        H    = chrom_fmt[0]
        O    = chrom_fmt[1]

        # ---- hidden layer ---------------------------------------------
        h = np.empty(H, dtype=np.float32)
        for j in range(H):
            acc = 0.0
            for i in range(N_in):
                acc += I[i] * chrom[idx]; idx += 1
            acc += chrom[N_in*H + j]
            h[j] = math.tanh(acc)

        idx = N_in*H + H
        out = np.empty(O, dtype=np.float32)
        for o in range(O):
            acc = 0.0
            for j in range(H):
                acc += h[j] * chrom[idx]; idx += 1
            acc += chrom[N_in*H + H + H*O + o]
            out[o] = math.tanh(acc)

        return out
    
    def print_chromosome(self, chrom: np.array):
        """
        Pretty-print the weights and biases from a chromosome for a feed-forward NN,
        with neuron inputs as rows and neuron outputs as columns.
        """
        idx = 0
        print("Input-Hidden Weights (W_in):")
        # W_in: shape (I, H)
        for i in range(self.I):
            row = []
            for h in range(self.H):
                row.append(f"{chrom[idx + h + i*self.H]: .4f}")
            print(f"  Input {i}: [{', '.join(row)}]")
        idx += self.I * self.H

        print("\nHidden Biases (b_h):")
        for h in range(self.H):
            print(f"  Hidden {h}: {chrom[idx]: .4f}")
            idx += 1

        print("\nHidden-Output Weights (W_out):")
        # W_out: shape (H, O)
        for h in range(self.H):
            row = []
            for o in range(self.O):
                row.append(f"{chrom[idx + o + h*self.O]: .4f}")
            print(f"  Hidden {h}: [{', '.join(row)}]")
        idx += self.H * self.O

        print("\nOutput Biases (b_out):")
        for o in range(self.O):
            print(f"  Output {o}: {chrom[idx]: .4f}")
            idx += 1
