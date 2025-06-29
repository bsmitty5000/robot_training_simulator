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
            #out[o] = math.tanh(acc)
            
            out[o] = 1.0 / (1.0 + math.exp(-acc))

        return out
    
    def print_chromosome(self, chrom: np.array):
        """
        Pretty-print the weights and biases from a chromosome for a feed-forward NN,
        with neuron inputs as rows and neuron outputs as columns.
        """
        weights = chrom[:-self.H - self.O]  # all but last H + O
        biases  = chrom[-self.H - self.O:]  # last H + O
        
        # Input → Hidden
        print("// Weights from input to hidden ({}x{})".format(self.I, self.H))
        print("static const float w_input_hidden[{}][{}] = {{".format(self.I, self.H))
        for i in range(self.I):
            row = weights[i*self.H:(i+1)*self.H]
            print("    {" + ", ".join(f"{v:+0.4f}" for v in row) + "},")
        print("};\n")

        # Hidden biases
        print("// Biases for hidden layer ({})".format(self.H))
        b_hidden = biases[:self.H]
        print("static const float b_hidden[{}] = {{ {} }};\n".format(
            self.H, ", ".join(f"{v:+0.4f}" for v in b_hidden)
        ))

        # Hidden → Output
        print("// Weights from hidden to output ({}x{})".format(self.H, self.O))
        print("static const float w_hidden_output[{}][{}] = {{".format(self.H, self.O))
        for h in range(self.H):
            row = weights[self.I*self.H + h*self.O : self.I*self.H + (h+1)*self.O]
            print("    {" + ", ".join(f"{v:+0.4f}" for v in row) + "},")
        print("};\n")

        # Output biases
        print("// Biases for output layer ({})".format(self.O))
        b_output = biases[self.H:]
        print("static const float b_output[{}] = {{ {} }};".format(
            self.O, ", ".join(f"{v:+0.4f}" for v in b_output)
        ))
