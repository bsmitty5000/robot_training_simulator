import numpy as np

class NumpyMLP:
    def __init__(self, layer_sizes, seed=None):
        """
        A simple feed-forward MLP using NumPy.
        
        Args:
            layer_sizes (List[int]): e.g. [3, 5, 4, 2]
                - 3 inputs, two hidden layers (5 & 4 neurons), 2 outputs
            seed (int, optional): for reproducible weight initialization
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.layer_sizes = layer_sizes
        # initialize weights and biases with list comprehensions
        self.weights = [
            np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            for n_in, n_out in zip(layer_sizes, layer_sizes[1:])
        ]
        self.biases = [
            np.zeros(n_out)
            for n_out in layer_sizes[1:]
        ]

    def forward(self, x):
        """
        Forward pass through all layers with tanh activations.
        
        Args:
            x (np.ndarray): shape (input_size,)
        
        Returns:
            np.ndarray: shape (output_size,)
        """
        a = x
        for W, b in zip(self.weights, self.biases):
            a = np.tanh(a @ W + b)
        return a

if __name__ == "__main__":
    # Example: 3 inputs → 5 → 4 → 2 outputs
    net = NumpyMLP([3, 5, 4, 2], seed=42)

    # Print layer dimensions
    for idx, (W, b) in enumerate(zip(net.weights, net.biases), start=1):
        print(f"Layer {idx}: {W.shape[0]}→{W.shape[1]}, bias shape {b.shape}")

    # Sample forward pass
    inputs = np.array([0.6, -0.2, 0.1])
    outputs = net.forward(inputs)
    print("Input:", inputs)
    print("Output:", outputs)
