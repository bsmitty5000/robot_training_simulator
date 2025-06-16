import math
from typing import Sequence, List

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