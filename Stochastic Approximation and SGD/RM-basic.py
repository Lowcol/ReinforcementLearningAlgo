import numpy as np
import matplotlib.pyplot as plt

# True root is x* = 2
np.random.seed(0)
x=0.0  # initial guess of the root
alpha=0.1  # learning rate
history = []

for t in range(100):
    noise = np.random.randn()  * 0.5
    Y = x + noise - 2  # noisy observation of gradient
    x = x - alpha * Y # RM update
    history.append(x)

plt.plot(history, label='Estimate of x')
plt.axhline(2, color='r', linestyle='--', label='True root')
plt.xlabel('Iteration')
plt.ylabel('Estimation of x')
plt.legend()
plt.show()

'''
The code is a simple simulation of the 
Robbins–Monro stochastic approximation 
algorithm — a foundational idea behind 
how many learning algorithms (like Q-learning
or gradient descent) learn from noisy data.
'''