import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Assuming y_true and y_pred are your target and prediction tensors
y_true = Variable(torch.Tensor([1, 0, 0]))
y_pred = Variable(torch.Tensor([0.0, 0.0, 0.0]), requires_grad=True)

# Calculate the cross-entropy loss
a = y_pred.unsqueeze(0)
b = torch.argmax(y_true).unsqueeze(0)
loss = F.cross_entropy(a, b)


# Backward pass to calculate the gradient
loss.backward()
print(loss)

# Access the gradient
gradient = y_pred.grad
print(gradient)