import random
import matplotlib.pyplot as plt

# Generate 10,000 random integers between 1 and 10
data = [random.randint(1, 10) for _ in range(10000)]

# Plot the frequency distribution
plt.hist(data, bins=range(1, 12), align='left', rwidth=0.8)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Random Integers (1-10)')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set the mean and standard deviation for the normal distribution
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution

# Generate random numbers following a normal distribution
data = np.random.normal(mean, std_dev, 10000)  # Generate 10,000 samples

# Plot the histogram to visualize the normal distribution
plt.hist(data, bins=50, density=True, alpha=0.6, color='b')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.show()