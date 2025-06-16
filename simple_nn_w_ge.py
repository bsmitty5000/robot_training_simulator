import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
DT = 0.1              # Time step (s)
MAX_STEPS = 300       # Maximum steps per episode
ROBOT_RADIUS = 0.1    # Robot radius in meters
SENSOR_ANGLES = [np.pi/4, 0, -np.pi/4]  # Left, center, right
SENSOR_RANGE = 2.0    # Max sensor range (m)
WORLD_BOUNDS = [(-2, 2), (-2, 2)]  # x and y bounds

# Define the neural network architecture
class SimpleNN:
    def __init__(self, weights=None):
        self.input_size = 3
        self.hidden_size = 3
        self.output_size = 2
        self.genome_size = (self.input_size * self.hidden_size +
                            self.hidden_size * self.output_size +
                            self.hidden_size + self.output_size)
        self.weights = weights if weights is not None else np.random.uniform(-1, 1, self.genome_size)
        self.decode_weights()

    def decode_weights(self):
        idx = 0
        self.w1 = self.weights[idx:idx + self.input_size*self.hidden_size].reshape(self.input_size, self.hidden_size)
        idx += self.input_size*self.hidden_size
        self.w2 = self.weights[idx:idx + self.hidden_size*self.output_size].reshape(self.hidden_size, self.output_size)
        idx += self.hidden_size*self.output_size
        self.b1 = self.weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        self.b2 = self.weights[idx:idx + self.output_size]

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        o = np.tanh(h @ self.w2 + self.b2)
        # Outputs: [throttle, steering]
        return o

# Robot class
class Robot:
    def __init__(self, nn):
        self.nn = nn
        self.reset()

    def reset(self):
        self.x, self.y = 0.0, 0.0
        self.theta = 0.0
        self.alive = True
        self.time_alive = 0.0
        self.collision = False
        self.history = []

    def sense(self):
        readings = []
        for angle in SENSOR_ANGLES:
            ray_angle = self.theta + angle
            for d in np.linspace(0, SENSOR_RANGE, 100):
                rx = self.x + d * np.cos(ray_angle)
                ry = self.y + d * np.sin(ray_angle)
                # check world bounds
                if not (WORLD_BOUNDS[0][0] <= rx <= WORLD_BOUNDS[0][1] and
                        WORLD_BOUNDS[1][0] <= ry <= WORLD_BOUNDS[1][1]):
                    readings.append(d)
                    break
            else:
                readings.append(SENSOR_RANGE)
        return np.array(readings) / SENSOR_RANGE  # normalize

    def step(self, dt):
        # sense
        sensors = self.sense()
        # decide
        throttle, steer = self.nn.forward(sensors)
        v = throttle * 0.5   # max speed ~0.5 m/s
        omega = steer * 1.0  # max turn rate ~1 rad/s
        # move
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        # check collision with walls
        if not (WORLD_BOUNDS[0][0] + ROBOT_RADIUS <= self.x <= WORLD_BOUNDS[0][1] - ROBOT_RADIUS and
                WORLD_BOUNDS[1][0] + ROBOT_RADIUS <= self.y <= WORLD_BOUNDS[1][1] - ROBOT_RADIUS):
            self.alive = False
            self.collision = True
        # log
        self.time_alive += dt
        self.history.append((self.x, self.y, sensors, throttle, steer))

# Fitness function (easily modifiable)
def compute_fitness(robot):
    # Basic: time before collision
    fitness = robot.time_alive
    # Example extensions (uncomment to use):
    # distance_traveled = sum(np.hypot(dx, dy) for (dx, dy, *_), (px, py, *_) in zip(robot.history[1:], robot.history[:-1]))
    # fitness += 0.5 * distance_traveled  # reward forward movement
    # fitness -= 2 * int(robot.collision) # penalty for collision
    return fitness

# Evaluate a genome
def evaluate_genome(weights):
    nn = SimpleNN(weights)
    robot = Robot(nn)
    for _ in range(MAX_STEPS):
        robot.step(DT)
        if not robot.alive:
            break
    fitness = compute_fitness(robot)
    return fitness, robot.history

# Example usage
if __name__ == "__main__":
    # Random genome test
    genome = np.random.uniform(-1, 1, SimpleNN().genome_size)
    fitness, history = evaluate_genome(genome)
    print(f"Fitness: {fitness:.2f}")
    
    # Plot trajectory
    xs, ys = zip(*[(h[0], h[1]) for h in history])
    plt.figure(figsize=(5,5))
    plt.plot(xs, ys, '-o', markersize=2)
    # Draw world bounds
    plt.xlim(WORLD_BOUNDS[0])
    plt.ylim(WORLD_BOUNDS[1])
    plt.title("Robot Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()
