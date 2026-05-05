import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------


class Linear():
    def __init__(self, in_features: int, out_features: int, batch_size: int, lr=0.1):
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.weight = np.random.normal(
            size=(in_features, out_features)) * np.sqrt(1. / in_features)
        self.bias = np.random.normal(
            size=(out_features,)) * np.sqrt(1. / in_features)
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))

    # Forward prop: computes the layer activation x_l = W_l x_{l-1} + b_l
    def forward(self, input):
        # TODO: Implement the forward pass
        self.input = input

        output = input @ self.weight + self.bias
        return output

    # Back prop: uses the upstream gradient and the chain rule to compute gradients for W, b, and x_{l-1}.
    def backward(self, grad_output):
        # TODO: Implement the backward pass
        grad_input = grad_output @ self.weight.T
        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_input

    # Weight update: applies gradient descent W := W - eta * grad_W and b := b - eta * grad_b
    def update(self):
        # TODO: Implement the parameter update step (using self.lr)
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias

# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------


class Sigmoid():
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = np.zeros((batch_size, in_features))
        self.output = np.zeros((batch_size, in_features))

    # Forward prop: applies the non-linearity f, here sigmoid sigma(x) = 1 / (1 + exp(-x)).
    def forward(self, input):
        # TODO: Implement the forward pass
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    # Back prop: multiplies the upstream gradient with the local derivative sigma'(x) = sigma(x)(1 - sigma(x)).
    def backward(self, grad_output):
        # TODO: Implement the backward pass
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

# ---------------------------------------------------------
# Utilities for training
# ---------------------------------------------------------


def Softmax(input):
    # TODO: Implement the softmax function
    shifted = input - np.max(input, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return output


def compute_loss(target, prediction):
    ''' Computes the cross-entropy loss '''
    eps = 1e-12
    prediction = np.clip(prediction, eps, 1.0 - eps)
    return -np.sum(target * np.log(prediction)) / prediction.shape[0]


def compute_gradient(target, prediction):
    ''' 
    Computes the gradient of the cross-entropy loss w.r.t. the predictions.
    The below formula is valid for softmax + cross-entropy loss with one-hot targets.
    Due to this, we do not need to implement a backward pass for the softmax layer.
    '''
    return (prediction - target) / prediction.shape[0]


def one_hot(a, num_classes):
    ''' Converts an integer array to a one-hot encoded array '''
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class MLP():
    ''' A simple 3-layer MLP '''

    def __init__(self, batch_size, lr):
        super(MLP, self).__init__()
        self.linear0 = Linear(28*28, 512, batch_size, lr)
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
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.numpy(), target.numpy()

        output = model.forward(data)
        loss = compute_loss(target, output)
        gradient = compute_gradient(target, output)

        model.backward(gradient)
        model.update()

        losses.append(loss)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

    return float(np.mean(losses))


def test(args, model, test_loader, epoch):
    test_losses = []
    correct = 0

    for data, target in test_loader:
        data, target = data.numpy(), target.numpy()

        output = model.forward(data)

        # get the index of the max log-probability
        pred = output.argmax(axis=1, keepdims=True)
        correct += np.equal(pred.squeeze(), target).sum()

        # Convert to 1-hot encoding
        target = one_hot(target, 10)
        loss = compute_loss(target, output)
        test_losses.append(loss)

    test_loss = float(np.mean(test_losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, float(accuracy)


# This function plots the training loss, test loss, and test accuracy.
# These plots help us see whether the model improves during training.
def plot_default_training(train_losses, test_losses, test_accuracies, epochs):
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot 1: Train and test loss
    # Expected:
    # We expected a decrease in the training loss over the epochs, as the weights are updated after every batch
    # from the model. Also we expected the test loss to decrease if the model generalizes well to the unseen MNIST images.
    # Observed:
    # We observed that the training loss decreases during training -> the model is learning from the training data.
    # The test loss also decreases, mostly in the first epochs. A slower decrease in the test loss means that the
    # model has already learned most of the useful patterns.
    plt.savefig("train_test_loss.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot 2: Test Accuracy
    # Expected:
    # We expected the test accuracy to increase over time, as lower loss means better performance regarding classification.
    # Observed:
    # An increase in the test accuracy over the epochs is observed. The model correctly classifies more MNIST digits as training continues.
    # Over the epochs there is a smaller improvement, as the model is close to its best performance.
    plt.savefig("test_accuracy.png")
    plt.show()


# Function which trains the model several times with different learning rates.
# A new model is initialized for each learning rate and is trained.
# We compare the results using test loss and test accuracy plots.
def investigate_learning_rates(args, train_loader, test_loader):
    learning_rates = [1.0, 0.2, 0.05, 0.01, 0.001]

    all_test_losses = {}
    all_test_accuracies = {}

    original_lr = args.lr

    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}")

        args.lr = lr
        np.random.seed(args.seed)
        model = MLP(args.batch_size, args.lr)

        test_losses = []
        test_accuracies = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, train_loader, epoch)
            test_loss, test_accuracy = test(args, model, test_loader, epoch)

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(
                f"LR={lr} | Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Test Loss={test_loss:.4f}, "
                f"Test Accuracy={test_accuracy:.2f}%"
            )

        all_test_losses[lr] = test_losses
        all_test_accuracies[lr] = test_accuracies

    args.lr = original_lr
    epochs = range(1, args.epochs + 1)

    plt.figure()
    for lr in learning_rates:
        plt.plot(epochs, all_test_losses[lr], label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot 3: Test loss for different learning rates
    # Expected:
    # Small learning rates like 0.001 were expected to learn slowly and ones likes 1.0 to be little unstable.
    # Medium learning rates like 0.05 and 0.01 should give more stable results.
    # Observed:
    # Learning rate has a big influence on the test loss. Small learning rates reduce the loss slowly.
    # Large learning rates could improve faster, but are less stable. Medium learning rates give the most stable results.
    plt.savefig("different_lr_test_loss.png")
    plt.show()

    plt.figure()
    for lr in learning_rates:
        plt.plot(epochs, all_test_accuracies[lr], label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot 4: Test accuracy for different learning rates
    # Expected:
    # We expected a good learning rate to reach high accuracy faster. Slow improval in small learning rates and large ones can fluctuate.
    # Observed:
    # We observed that different learning rates -> different training behavior.Small learning rates improve slowly and large lr more quickly, but possibly unstable.
    # We concluded that a medium learning rate gives most reliable accuracy.
    plt.savefig("different_lr_test_accuracy.png")
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (epochs = 30)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform,
                                   target_transform=torchvision.transforms.Compose([
                                       lambda x: np.array([x]),
                                       lambda x: one_hot(x, 10),
                                       lambda x: np.squeeze(x),
                                   ]))

    dataset_test = datasets.MNIST('../data', train=False, download=True,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=args.batch_size)

    begin_time = time.time()

    model = MLP(args.batch_size, args.lr)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, train_loader, epoch)
        test_loss, test_accuracy = test(args, model, test_loader, epoch)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    total_time = time.time() - begin_time
    print(f"\nTraining time: {total_time:.1f}s")

    epochs = range(1, args.epochs + 1)

    # For the plots, modify the train and test functions to return the losses and accuracies
    # Create the required plotting functions and call them here
    plot_default_training(train_losses, test_losses, test_accuracies, epochs)

    investigate_learning_rates(args, train_loader, test_loader)


if __name__ == '__main__':
    main()
