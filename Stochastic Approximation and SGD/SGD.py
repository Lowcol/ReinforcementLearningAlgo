import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

theta = 0.0  # initial parameter
alpha = 0.1  # learning rate
history = []

for t in range(100):
    # Gradient of (θ - 5)^2 is 2*(θ - 5)
    # Add some random noise to simulate stochastic gradient
    grad = 2 * (theta - 5) + np.random.randn()  * 0.5
    theta = theta - alpha * grad  # SGD update
    history.append(theta)
    
plt.plot(history, label='Estimate of θ')
plt.axhline(5, color='r', linestyle='--', label='True minimum')
plt.xlabel('Iteration')
plt.ylabel('Estimation of θ')
plt.legend()
plt.show()

