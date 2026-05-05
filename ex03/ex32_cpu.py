import argparse
import time
import numpy as np
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms


class Linear:
    """A fully connected layer implemented with NumPy arrays."""

    def __init__(self, in_features: int, out_features: int, batch_size: int, lr: float = 0.1) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.weight = np.random.normal(
            size=(in_features, out_features)) * np.sqrt(1.0 / in_features)
        self.bias = np.random.normal(
            size=(out_features,)) * np.sqrt(1.0 / in_features)
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output @ np.transpose(self.weight)
        self.grad_weight = np.transpose(self.input) @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_input

    def update(self) -> None:
        self.weight = self.weight - self.lr * self.grad_weight / self.batch_size
        self.bias = self.bias - self.lr * self.grad_bias / self.batch_size


class Sigmoid:
    def __init__(self, in_features: int, batch_size: int) -> None:
        self.input = np.zeros(batch_size)
        self.output = np.zeros(batch_size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.output * (1 - self.output))


def Softmax(input: np.ndarray) -> np.ndarray:
    output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
    return output


def compute_loss(target: np.ndarray, prediction: np.ndarray) -> float:
    return -np.sum(target * np.log(prediction)) / prediction.shape[0]


def compute_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return prediction - target


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class MLP:
    def __init__(self, batch_size: int, lr: float) -> None:
        self.linear0 = Linear(28 * 28, 512, batch_size, lr)
        self.sigmoid0 = Sigmoid(512, batch_size)
        self.linear1 = Linear(512, 128, batch_size, lr)
        self.sigmoid1 = Sigmoid(128, batch_size)
        self.linear2 = Linear(128, 10, batch_size, lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        x = Softmax(x)
        return x

    def backward(self, x: np.ndarray) -> None:
        x = self.linear2.backward(x)
        x = self.sigmoid1.backward(x)
        x = self.linear1.backward(x)
        x = self.sigmoid0.backward(x)
        x = self.linear0.backward(x)

    def update(self) -> None:
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()


def train(args: argparse.Namespace, model: MLP, train_loader: torch.utils.data.DataLoader, epoch: int) -> float:
    losses = []

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        leave=False,
        position=1,
    )

    for batch_idx, (data, target) in pbar:
        data = data.numpy()
        target = target.numpy()

        output = model.forward(data)
        loss = compute_loss(target, output)
        gradient = compute_gradient(target, output)

        model.backward(gradient)
        model.update()

        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({"loss": float(loss)})

        losses.append(float(loss / data.shape[0]))

    return np.mean(losses)


def test(model: MLP, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()

        output = model.forward(data)
        pred = output.argmax(axis=1, keepdims=True)

        correct += np.equal(pred.squeeze(), target).sum()

        target = one_hot(target, 10)
        loss = compute_loss(target, output)
        test_loss += loss

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return float(test_loss), float(accuracy)


def plot_losses(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    test_accuracies: Sequence[float],
) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training and Test Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics_cpu.pdf")
    plt.close()


def plot_accuracy_over_time(times: Sequence[float], test_accuracies: Sequence[float]) -> None:
    plt.figure()
    plt.plot(times, test_accuracies, label="CPU")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy over Time (CPU)")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_time_cpu.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy MNIST MLP")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset_train = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=torchvision.transforms.Compose(
            [
                lambda x: np.array([x]),
                lambda x: one_hot(x, 10),
                lambda x: np.squeeze(x),
            ]
        ),
    )

    dataset_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=args.batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=args.batch_size
    )

    with torch.no_grad():
        model = MLP(args.batch_size, args.lr)

        train_losses = []
        test_losses = []
        test_accuracies = []
        times = []

        start_time = time.time()

        pbar = tqdm(range(1, args.epochs + 1),
                    desc="Training Progress", position=0)

        for epoch in pbar:
            epoch_train_loss = train(args, model, train_loader, epoch)
            epoch_test_loss, epoch_acc = test(model, test_loader)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_acc)

            pbar.set_postfix(
                {
                    "Test Loss": epoch_test_loss,
                    "Test Acc (%)": epoch_acc,
                    "Time (s)": elapsed_time,
                }
            )

    total_time = time.time() - start_time
    print(f"Total CPU training time: {total_time:.2f} seconds")
    print(f"Final CPU test accuracy: {test_accuracies[-1]:.2f}%")

    plot_losses(train_losses, test_losses, test_accuracies)
    plot_accuracy_over_time(times, test_accuracies)

    np.savez(
        "cpu_results.npz",
        times=np.array(times),
        test_accuracies=np.array(test_accuracies),
        train_losses=np.array(train_losses),
        test_losses=np.array(test_losses),
    )


if __name__ == "__main__":
    main()
