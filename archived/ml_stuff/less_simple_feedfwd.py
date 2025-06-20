import numpy as np

# -------------------------------
# More Flexible MLP with Matrices
# -------------------------------

class MultiLayerNN:
    def __init__(self, layer_sizes, seed=None):
        """
        layer_sizes: list of ints, e.g. [3, 5, 4, 2]
          - 3 inputs, two hidden layers (5 and 4 neurons), and 2 outputs
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights & biases as lists of numpy arrays
        # weights[i] has shape (layer_sizes[i], layer_sizes[i+1])
        # biases[i] has shape (layer_sizes[i+1],)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            # He initialization for weights
            W = np.random.randn(in_size, out_size) * np.sqrt(2 / in_size)
            b = np.zeros(out_size)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """
        Forward pass through all layers using tanh activation.
        x: 1D numpy array of length layer_sizes[0]
        returns: 1D numpy array of length layer_sizes[-1]
        """
        a = x
        for W, b in zip(self.weights, self.biases):
            z = a @ W + b      # Linear step
            a = np.tanh(z)      # Nonlinear activation
        return a

if __name__ == "__main__":
    # Example: 3 inputs → 5 neurons → 4 neurons → 2 outputs
    layer_sizes = [3, 5, 4, 2]
    net = MultiLayerNN(layer_sizes, seed=42)

    # Show weight & bias shapes for each layer
    print("Network architecture:")
    for i, (W, b) in enumerate(zip(net.weights, net.biases)):
        print(f" Layer {i}:")
        print(f"  weights shape = {W.shape}")
        print(f"  biases shape  = {b.shape}")

    # Run a sample forward pass
    x = np.array([0.6, -0.2, 0.1])  # example normalized sensor inputs
    output = net.forward(x)
    print("\nSample input:", x)
    print("Network output:", output)
