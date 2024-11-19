import numpy as np
import matplotlib.pyplot as plt

# Define the hyperplane parameters (example values)
w = np.array([1, -2])  # normal vector
b = -3                # bias term

# Function to calculate the projection onto the hyperplane
def project_onto_hyperplane(x, w, b):
    return x - ((np.dot(w, x) + b) / np.dot(w, w)) * w

# Generate random point for illustration
x_point = np.array([2, 3])

# Project the point onto the hyperplane
x0_point = project_onto_hyperplane(x_point, w, b)

# Plotting
plt.figure(figsize=(8, 8))

# Plot the hyperplane
xx, yy = np.meshgrid(range(-5, 6), range(-5, 6))
zz = (-w[0] * xx - w[1] * yy - b) / w[1]
plt.contour(xx, yy, zz, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'], linewidths=2)

# Plot the original point and its projection
plt.scatter(x_point[0], x_point[1], marker='o', label='Original Point')
plt.scatter(x0_point[0], x0_point[1], marker='x', label='Projected Point')

# Draw a line connecting the original and projected points
plt.plot([x_point[0], x0_point[0]], [x_point[1], x0_point[1]], 'r--')

# Set labels and legend
plt.title('Projection onto SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()