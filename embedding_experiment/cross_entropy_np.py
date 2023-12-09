import numpy as np

# Assuming y_true and y_pred are your target and prediction arrays
y_true = np.array([1, 0, 0])
y_pred = np.array([1.0, 0.0, 0.0])
y_pred = y_pred.astype(np.float32)  # Ensure data type consistency

# Calculate the cross-entropy loss
a = np.expand_dims(y_pred, axis=0)
b = np.argmax(y_true)
loss = -np.sum(y_true * np.log(a + 1e-10))  # Adding a small epsilon to avoid log(0)

# Backward pass to calculate the gradient
gradient = -(y_true / (a + 1e-10))

# Access the gradient
print(gradient)
