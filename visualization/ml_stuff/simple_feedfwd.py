import numpy as np

# -----------------------------
# Beginner-Friendly Neural Net
# -----------------------------

# 1) Define the network size
input_size = 3    # 3 input sensors (e.g., left, center, right distances)
hidden_size = 3   # 3 neurons in the hidden layer
output_size = 2   # 2 outputs (e.g., left motor PWM, right motor PWM)

# 2) Initialize weights and biases
# Weights connect one layer to the next; biases shift the activation
np.random.seed(0)  # ensures repeatable "random" numbers for learning/testing
W1 = np.random.randn(input_size, hidden_size) * 0.1   # input → hidden
b1 = np.zeros(hidden_size)                            # hidden layer biases
W2 = np.random.randn(hidden_size, output_size) * 0.1  # hidden → output
b2 = np.zeros(output_size)                            # output layer biases

# 3) Activation function
def tanh(x):
    """Apply the tanh activation elementwise."""
    return np.tanh(x)

# 4) Forward pass function
def forward_pass(inputs):
    """
    Compute one pass through the network.
    
    inputs: numpy array of length input_size
    returns: numpy array of length output_size
    """
    # Hidden layer computation
    # Step A: Weighted sum of inputs + bias
    hidden_raw = np.dot(inputs, W1) + b1
    # Step B: Nonlinear activation
    hidden_activated = tanh(hidden_raw)

    # Output layer computation
    output_raw = np.dot(hidden_activated, W2) + b2
    output_activated = tanh(output_raw)

    return output_activated

# 5) Example usage
if __name__ == "__main__":
    # Example sensor readings (normalized between -1 and 1 or 0 and 1)
    sensors = np.array([0.5, 0.8, 0.2])
    
    # Run the forward pass
    outputs = forward_pass(sensors)
    
    # Interpret outputs
    # Since outputs are in [-1,1], map to PWM if needed:
    # PWM_left  = int((outputs[0] + 1)/2 * 255)
    # PWM_right = int((outputs[1] + 1)/2 * 255)
    
    print("Raw network outputs (range -1 to 1):", outputs)
    # Uncomment below to see PWM mapping
    # print("Mapped to PWM (0–255):", PWM_left, PWM_right)
