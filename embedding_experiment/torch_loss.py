import torch
import torch.nn.functional as F
from torch.autograd import Variable

def torch_cross_entropy_loss_and_gradient(y_true, y_pred):
    # Assuming y_true and y_pred are your target and prediction tensors
    y_true = Variable(torch.Tensor(y_true))
    y_pred = Variable(torch.Tensor(y_pred), requires_grad=True)

    # Calculate the cross-entropy loss
    a = y_pred.unsqueeze(0)
    b = torch.argmax(y_true).unsqueeze(0)
    loss = F.cross_entropy(a, b)

    # Backward pass to calculate the gradient
    loss.backward()

    # Access the gradient
    gradient = y_pred.grad.numpy()

    return loss.item(), gradient

if __name__ == '__main__':
    y_true = [1, 0, 0]
    y_pred = [0.0, 0.0, 0.0]
    loss, gradient = torch_cross_entropy_loss_and_gradient(y_true, y_pred)
    print(loss)
    print(gradient)