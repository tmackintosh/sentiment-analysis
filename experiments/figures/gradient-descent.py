import matplotlib.pyplot as plt
import numpy as np

# Define your cost function
def cost_function(x, y):
    return x**2 + y**2  # Example: a simple quadratic cost function

# Define the gradient of the cost function
def gradient(x, y):
    return np.array([2*x, 2*y])

# Gradient Descent function
def gradient_descent(starting_point, learning_rate, iterations):
    trajectory = []
    current_point = starting_point
    
    for _ in range(iterations):
        trajectory.append(current_point)
        current_point = current_point - learning_rate * gradient(*current_point)
    
    return np.array(trajectory)

# Generate data for the plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = cost_function(X, Y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Perform gradient descent
starting_point = np.array([4.5, 4.5])  # Initial guess
learning_rate = 0.1
iterations = 50
trajectory = gradient_descent(starting_point, learning_rate, iterations)

# Plot the trajectory on the cost function surface
ax.plot(trajectory[:, 0], trajectory[:, 1], cost_function(*trajectory.T), color='red', marker='o')

# Add labels
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Cost')

# Show the plot
plt.show()