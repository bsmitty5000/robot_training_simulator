from typing import Any, Sequence, List

import numpy as np
from ml_stuff.decision_base import DecisionBase

class FFNetDecisionMaker(DecisionBase):
    def __init__(self, layer_sizes: list[int]):
        """
        layer_sizes: e.g., [3, 4, 2] for 3 inputs, 1 hidden layer of 4, 2 outputs
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        # Initialize weights and biases as empty lists
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

    def set_weights(self, weights: list[list[list[float]]], biases: list[list[float]]) -> None:
        """
        weights: list of weight matrices, one per layer (shape: [out][in])
        biases: list of bias vectors, one per layer (shape: [out])
        """
        assert len(weights) == self.num_layers
        assert len(biases) == self.num_layers
        self.weights = weights
        self.biases = biases

    @property
    def output_size(self) -> int:
        return self.layer_sizes[-1]

    def decide(self, sensor_distances: Sequence[float]) -> List[float]:
        a = list(sensor_distances)
        for W, b in zip(self.weights, self.biases):
            a = np.tanh(a @ W + b)
            
        return a
    
    @staticmethod
    def encode(*args, **kwargs) -> List[Any]:

        weights = []
        biases = []
        if 'weights' in kwargs:
            weights = kwargs['weights']
        if 'biases' in kwargs:
            biases = kwargs['biases']
        genome = []
        for W, b in zip(weights, biases):
            genome.extend(W.flatten())  # Flatten weights first
            genome.extend(b.flatten())  # Then flatten biases

        return list(genome)
    
    @staticmethod
    def from_genotype(genotype: List[Any], *args, **kwargs) -> Any:
        
        weights: list[list[list[float]]] = []
        biases: list[list[float]] = []
        layer_sizes: list[int] = []
        idx = 0

        if 'layer_sizes' in kwargs:
                layer_sizes = kwargs['layer_sizes']

        num_layers = len(layer_sizes) - 1

        for l in range(num_layers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l + 1]
            weights_len = layer_sizes[l] * layer_sizes[l + 1]
            biases_len = layer_sizes[l + 1]

            W_flat = genotype[idx:idx + weights_len]
            b_flat = genotype[idx + weights_len:idx + weights_len + biases_len]

            idx += (weights_len + biases_len)
            
            W = np.array(W_flat).reshape((n_in, n_out))
            b = np.array(b_flat)
            
            weights.append(W)
            biases.append(b)

        return weights, biases

    def print_info(self) -> str:
        return f"genotype: {FFNetDecisionMaker.encode(weights=self.weights, biases=self.biases)}"