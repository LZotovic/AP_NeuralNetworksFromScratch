import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms


hidden_sizes = [128, 256, 512, 1024, 1534, 2048, 3072]
batch_size = 64
lr = 0.003


def one_hot_np(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def softmax(x, xp):
    return xp.exp(x) / xp.sum(xp.exp(x), axis=1, keepdims=True)


class Linear:
    def __init__(self, in_features, out_features, batch_size, lr, xp):
        self.xp = xp
        self.batch_size = batch_size
        self.lr = lr
        self.weight = xp.random.normal(size=(in_features, out_features)) * xp.sqrt(1.0 / in_features)
        self.bias = xp.random.normal(size=(out_features,)) * xp.sqrt(1.0 / in_features)
        self.input = None

    def forward(self, x):
        self.input = x
        return x @ self.weight + self.bias

    def backward(self, grad_output):
        grad_input = grad_output @ self.xp.transpose(self.weight)
        self.grad_weight = self.xp.transpose(self.input) @ grad_output
        self.grad_bias = self.xp.sum(grad_output, axis=0)
        return grad_input

    def update(self):
        self.weight = self.weight - self.lr * self.grad_weight / self.batch_size
        self.bias = self.bias - self.lr * self.grad_bias / self.batch_size


class Sigmoid:
    def __init__(self, xp):
        self.xp = xp
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + self.xp.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class MLP:
    def __init__(self, hidden_size, batch_size, lr, xp):
        self.xp = xp
        self.linear0 = Linear(28 * 28, hidden_size, batch_size, lr, xp)
        self.sigmoid0 = Sigmoid(xp)
        self.linear1 = Linear(hidden_size, hidden_size, batch_size, lr, xp)
        self.sigmoid1 = Sigmoid(xp)
        self.linear2 = Linear(hidden_size, 10, batch_size, lr, xp)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        return softmax(x, self.xp)

    def backward(self, grad):
        grad = self.linear2.backward(grad)
        grad = self.sigmoid1.backward(grad)
        grad = self.linear1.backward(grad)
        grad = self.sigmoid0.backward(grad)
        grad = self.linear0.backward(grad)

    def update(self):
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()


def train_one_epoch(model, train_loader, xp):
    for data, target in train_loader:
        if xp == cp:
            data = cp.asarray(data.numpy())
            target = cp.asarray(target.numpy())
        else:
            data = data.numpy()
            target = target.numpy()

        output = model.forward(data)
        grad = output - target

        model.backward(grad)
        model.update()


def measure_time(hidden_size, train_loader, xp):
    print(f"  → Starting {'GPU' if xp == cp else 'CPU'} for size {hidden_size}", flush=True)

    model = MLP(hidden_size, batch_size, lr, xp)

    if xp == cp:
        cp.cuda.Stream.null.synchronize()

    start_time = time.time()
    train_one_epoch(model, train_loader, xp)

    if xp == cp:
        cp.cuda.Stream.null.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"  → Finished {'GPU' if xp == cp else 'CPU'} size {hidden_size} in {elapsed:.2f}s", flush=True)

    return elapsed


def main():
    print("Starting benchmark...", flush=True)

    torch.manual_seed(1)
    np.random.seed(1)
    cp.random.seed(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=torchvision.transforms.Compose([
            lambda x: np.array([x]),
            lambda x: one_hot_np(x, 10),
            lambda x: np.squeeze(x),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size
    )

    cpu_times = []
    gpu_times = []

    for hidden_size in hidden_sizes:
        print(f"\n===== Measuring hidden size {hidden_size} =====", flush=True)

        cpu_time = measure_time(hidden_size, train_loader, np)
        gpu_time = measure_time(hidden_size, train_loader, cp)

        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

        print(f"RESULT → hidden={hidden_size} | CPU={cpu_time:.2f}s | GPU={gpu_time:.2f}s", flush=True)

    cpu_times = np.array(cpu_times)
    gpu_times = np.array(gpu_times)
    speedups = cpu_times / gpu_times

    np.savez(
        "size_benchmark_results.npz",
        hidden_sizes=np.array(hidden_sizes),
        cpu_times=cpu_times,
        gpu_times=gpu_times,
        speedups=speedups,
    )

    # plot 1
    plt.figure()
    plt.plot(hidden_sizes, cpu_times, label="CPU")
    plt.plot(hidden_sizes, gpu_times, label="GPU")
    plt.xlabel("Hidden layer size")
    plt.ylabel("Time per epoch (seconds)")
    plt.title("CPU vs GPU Time per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_per_epoch_sizes.pdf")
    plt.close()

    # plot 2
    plt.figure()
    plt.plot(hidden_sizes, speedups, label="GPU speedup")
    plt.xlabel("Hidden layer size")
    plt.ylabel("Speedup (CPU/GPU)")
    plt.title("GPU Speedup")
    plt.legend()
    plt.grid(True)
    plt.savefig("gpu_speedup_sizes.pdf")
    plt.close()

    print("\nFINISHED BENCHMARK", flush=True)
    print("Sizes:", hidden_sizes, flush=True)
    print("CPU times:", cpu_times, flush=True)
    print("GPU times:", gpu_times, flush=True)
    print("Speedups:", speedups, flush=True)


if __name__ == "__main__":
    main()
