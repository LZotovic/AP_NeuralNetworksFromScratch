import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms


# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    def __init__(self, in_features: int, out_features: int, batch_size: int, lr=0.1):
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.weight = np.random.normal(size=(in_features, out_features)) * np.sqrt(1.0 / in_features)
        self.bias = np.random.normal(size=(out_features,)) * np.sqrt(1.0 / in_features)
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))

    def forward(self, input):
        # TODO: Implement the forward pass
        # self.input =
        # output =
        # return output
        raise NotImplementedError

    def backward(self, grad_output):
        # TODO: Implement the backward pass
        # grad_input =
        # self.grad_weight =
        # self.grad_bias =
        # return grad_input
        raise NotImplementedError

    def update(self):
        # TODO: Implement the parameter update step (using self.lr)
        # self.weight =
        # self.bias =
        raise NotImplementedError


# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = np.zeros(batch_size)

    def forward(self, input):
        self.input = input
        # TODO: Implement the forward pass
        # output =
        # return output
        raise NotImplementedError

    def backward(self, grad_output):
        # TODO: Implement the backward pass
        # grad_input =
        # return grad_input
        raise NotImplementedError


# ---------------------------------------------------------
# Utilities for training
# ---------------------------------------------------------
def Softmax(input):
    # TODO: Implement the softmax function
    # output =
    # return output
    raise NotImplementedError


def compute_loss(target, prediction):
    """Computes the cross-entropy loss"""
    return -np.sum(target * np.log(prediction)) / prediction.shape[0]


def compute_gradient(target, prediction):
    """
    Computes the gradient of the cross-entropy loss w.r.t. the predictions.
    The below formula is valid for softmax + cross-entropy loss with one-hot targets.
    Due to this, we do not need to implement a backward pass for the softmax layer.
    """
    return prediction - target


def one_hot(a, num_classes):
    """Converts an integer array to a one-hot encoded array"""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class MLP:
    """A simple 3-layer MLP"""

    def __init__(self, batch_size, lr):
        super(MLP, self).__init__()
        self.linear0 = Linear(28 * 28, 512, batch_size, lr)
        self.sigmoid0 = Sigmoid(512, batch_size)
        self.linear1 = Linear(512, 128, batch_size, lr)
        self.sigmoid1 = Sigmoid(128, batch_size)
        self.linear2 = Linear(128, 10, batch_size, lr)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        x = Softmax(x)
        return x

    def backward(self, x):
        x = self.linear2.backward(x)
        x = self.sigmoid1.backward(x)
        x = self.linear1.backward(x)
        x = self.sigmoid0.backward(x)
        x = self.linear0.backward(x)

    def update(self):
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()


def train(args, model, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.numpy(), target.numpy()
        output = model.forward(data)
        loss = compute_loss(target, output)
        gradient = compute_gradient(target, output)
        model.backward(gradient)
        model.update()
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / data.shape[0],
                )
            )


def test(args, model, test_loader, epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.numpy(), target.numpy()
        output = model.forward(data)
        pred = output.argmax(axis=1, keepdims=True)  # get the index of the max log-probability
        correct += np.equal(pred.squeeze(), target).sum()

        # Convert to 1-hot encoding
        target = one_hot(target, 10)
        loss = compute_loss(target, output)
        test_loss += loss

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transform,
        target_transform=torchvision.transforms.Compose([lambda x: np.array([x]), lambda x: one_hot(x, 10), lambda x: np.squeeze(x)]),
    )

    dataset_test = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size)

    with torch.no_grad():
        model = MLP(args.batch_size, args.lr)
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, epoch)
            test(args, model, test_loader, epoch)

    # For the plots, modify the train and test functions to return the losses and accuracies
    # Create the required plotting functions and call them here


if __name__ == '__main__':
    main()
