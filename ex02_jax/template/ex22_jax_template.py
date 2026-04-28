import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from typing import NamedTuple, Tuple, List
import functools
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # must before jax be imported

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class LinearParams(NamedTuple):
    weight: jnp.ndarray  # (in_features, out_features)
    bias: jnp.ndarray  # (out_features,)


class LinearGrads(NamedTuple):
    grad_weight: jnp.ndarray  # (in_features, out_features)
    grad_bias: jnp.ndarray  # (out_features,)


def linear_init(key: jax.Array, in_features: int, out_features: int) -> LinearParams:
    k_w, k_b = random.split(key)
    limit = jnp.sqrt(1.0 / in_features)
    weight = random.normal(k_w, (in_features, out_features)) * limit
    bias = random.normal(k_b, (out_features,)) * limit
    return LinearParams(weight=weight, bias=bias)

# Forward pass: computes y = xW + b and stores input for backprop
def linear_forward(params: LinearParams, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # TODO: Implement the forward pass
    output = x @ params.weight + params.bias
    return output, x # x as cache for backward

# Backward pass: computes gradients (dW, db, dX) using chain rule
def linear_backward(params: LinearParams, input: jnp.ndarray, grad_out: jnp.ndarray) -> Tuple[LinearGrads, jnp.ndarray]:
    # TODO: Implement the backward pass
    grad_input = grad_out @ params.weight.T
    grad_weight = input.T @ grad_out
    grad_bias = jnp.sum(grad_out, axis=0)
    return  LinearGrads(grad_weight=grad_weight, grad_bias=grad_bias), grad_input


def sgd_step(params: LinearParams, grads: LinearGrads, lr: float) -> LinearParams:
    return LinearParams(weight=params.weight - lr * grads.grad_weight, bias=params.bias - lr * grads.grad_bias)


def verify_linear_grads():
    key = random.PRNGKey(42)
    params = linear_init(key, in_features=4, out_features=8)
    x = random.normal(random.PRNGKey(1), (3, 4))  # batch=3

    out, cache = linear_forward(params, x)

    # assume loss = mean(y^2), then the gradient would be: 2y/N
    d_out = 2.0 * out / out.size

    manual_grads, manual_dx = linear_backward(params, cache, d_out)

    # the loss for jax.grad: mean(y^2)
    def loss_fn(params, x):
        y = x @ params.weight + params.bias
        return jnp.mean(y**2)

    auto_grads = jax.grad(loss_fn)(params, x)
    auto_dx = jax.grad(loss_fn, argnums=1)(params, x)

    # compare the grads
    err_w = jnp.max(jnp.abs(manual_grads.grad_weight - auto_grads.weight))
    err_b = jnp.max(jnp.abs(manual_grads.grad_bias - auto_grads.bias))
    err_dx = jnp.max(jnp.abs(manual_dx - auto_dx))

    tol = 1e-5
    ok = all(e < tol for e in [err_w, err_b, err_dx])
    print(f'\n  {"Linear backward is correct." if ok else "Something Wrong!"}')
    return ok


# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
# Forward pass: applies sigmoid activation σ(x) = 1 / (1 + e^-x)
def sigmoid_forward(x: jnp.ndarray) -> jnp.ndarray:
    # TODO: Implement the forward pass
    output =  1 / (1 + jnp.exp(-x))
    return output

# Backward pass: computes gradient using σ'(x) = σ(x)(1 - σ(x))
def sigmoid_backward(out: jnp.ndarray, grad_out: jnp.ndarray) -> jnp.ndarray:
    # TODO: Implement the backward pass
    grad_input = grad_out * out * (1 - out)
    return grad_input


def verify_sigmoid_grads():
    x = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
    grad_out = jnp.ones_like(x)

    # the loss for jax.grad
    def sigmoid_ref(x):
        return jnp.sum(1 / (1 + jnp.exp(-x)))

    grad_fn = jax.grad(sigmoid_ref)
    auto_grad = grad_fn(x)

    ### manual grad
    out = sigmoid_forward(x)
    manual_grads = sigmoid_backward(out, grad_out)

    # max_error = jnp.max(jnp.abs(auto_grad - manual_grads))
    ok = jnp.allclose(auto_grad, manual_grads, atol=1e-6)
    print(f'\n  {"Sigmoid backward is correct" if ok else "Something Wrong!"}')


# ---------------------------------------------------------
# Utilities for training
# ---------------------------------------------------------
# Computes stable softmax probabilities
def Softmax(input: jnp.ndarray) -> jnp.ndarray:
    """Compute the row-wise softmax of the input logits."""
    # TODO: Implement the softmax function
    shifted = input - jnp.max(input, axis=1, keepdims=True)
    exp_values = jnp.exp(shifted)
    output = exp_values / jnp.sum(exp_values, axis=1, keepdims=True)
    return output


def compute_loss(target, prediction):
    """Computes the cross-entropy loss"""
    return -jnp.sum(target * jnp.log(prediction)) / prediction.shape[0]


def compute_gradient(target, prediction):
    """
    Computes the gradient of the cross-entropy loss w.r.t. the predictions.
    The below formula is valid for softmax + cross-entropy loss with one-hot targets.
    Due to this, we do not need to implement a backward pass for the softmax layer.
    """
    return (prediction - target) / prediction.shape[0]


def one_hot(a, num_classes):
    """Converts an integer array to a one-hot encoded array"""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


# ---------------------------------------------------------
# MLP
# ---------------------------------------------------------

MLPParams = List[LinearParams]


def mlp_init(
    key: jax.Array,
    layer_sizes: List[int],  # e.g. [784, 512, 128, 10]
) -> MLPParams:
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        params.append(linear_init(subkey, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_forward(params: MLPParams, x: jnp.ndarray) -> Tuple[jnp.ndarray, list]:

    caches = []
    out = x

    for i, layer_params in enumerate(params):
        out, linear_cache = linear_forward(layer_params, out)
        # print(f"layer {i} | linear | W={layer_params.weight.shape}")
        if i < len(params) - 1:
            out = sigmoid_forward(out)
            caches.append((linear_cache, out))
            # print(f"layer {i} | sigmoid | out={out.shape}")
        else:
            caches.append((linear_cache, None))
            # print(f"Layer {i} | no act, last layer  | out={out.shape}")

    # softmax
    out = Softmax(out)
    # print(f"Output | softmax | out={out.shape}")
    return out, caches


def mlp_backward(params: MLPParams, caches: list, grad_out: jnp.ndarray) -> Tuple[List[LinearGrads], jnp.ndarray]:

    all_grads = []
    grad = grad_out
    for i in reversed(range(len(params))):
        linear_cache, sigmoid_cache = caches[i]

        if sigmoid_cache is not None:
            grad = sigmoid_backward(sigmoid_cache, grad)

        layer_grads, grad = linear_backward(params[i], linear_cache, grad)
        all_grads.append(layer_grads)

    all_grads.reverse()
    return all_grads, grad  # grad now is the grad of input


def mlp_sgd_step(params: MLPParams, grads: List[LinearGrads], lr: float) -> MLPParams:
    return [sgd_step(p, g, lr) for p, g in zip(params, grads)]


def verify_mlp_grads(train_loader):
    key = random.key(0)
    model_params = mlp_init(key, [784, 128, 10])
    x, y = next(iter(train_loader))
    x, y = jnp.array(x.numpy()), jnp.array(y.numpy())
    x = x.reshape(x.shape[0], -1)

    def loss_fn(model_params, x, y):
        output, _ = mlp_forward(model_params, x)
        return compute_loss(y, output)

    auto_grads = jax.grad(loss_fn, argnums=0)(model_params, x, y)

    output, caches = mlp_forward(model_params, x)
    gradient = compute_gradient(y, output)
    all_grads, _ = mlp_backward(model_params, caches, gradient)

    ### check the grads of the fist linear layer
    err_w = jnp.max(jnp.abs(all_grads[0].grad_weight - auto_grads[0].weight))
    err_b = jnp.max(jnp.abs(all_grads[0].grad_bias - auto_grads[0].bias))

    tol = 1e-5
    ok = all(e < tol for e in [err_w, err_b])
    print(f'\n  {"MLP backward is correct." if ok else "Something Wrong!"}')
    return ok


# ---------------------------------------------------------
# train and test
# ---------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(3,))  # Remove when debugging: jit suppresses print() calls after the first trace
def train_step(model_params, input, target, lr):
    output, caches = mlp_forward(model_params, input)
    loss = compute_loss(target, output)
    gradient = compute_gradient(target, output)
    all_grads, _ = mlp_backward(model_params, caches, gradient)
    model_params = mlp_sgd_step(model_params, all_grads, lr)
    return model_params, loss


def train_epoch(args, model_params, train_loader, epoch):
    """Train the model for a single epoch and return the average loss."""
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False, position=1)
    losses = []
    for batch_idx, (input, target) in pbar:
        input, target = jnp.array(input.numpy()), jnp.array(target.numpy())
        input = input.reshape(input.shape[0], -1)
        model_params, loss = train_step(model_params, input, target, args.lr)
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({'loss': loss})
        losses.append(loss)
    epoch_loss = jnp.mean(jnp.array(losses))
    return epoch_loss, model_params


@jax.jit  # Remove when debugging: jit suppresses print() calls after the first trace
def test_step(model_params, input, target):
    output, _ = mlp_forward(model_params, input)
    loss = compute_loss(target, output)
    correct = jnp.sum(jnp.argmax(output, axis=1) == jnp.argmax(target, axis=1))
    return loss, correct


def test_epoch(model_params, test_loader):
    test_loss = []
    correct = 0
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = jnp.array(input.numpy()), jnp.array(target.numpy())
        input = input.reshape(input.shape[0], -1)
        target = one_hot(target, 10)
        loss, _correct = test_step(model_params, input, target)
        test_loss.append(loss)
        correct += _correct
    accuracy = 100.0 * correct / len(test_loader.dataset)
    avg_loss = jnp.mean(jnp.array(test_loss))
    return accuracy, avg_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 14)')
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
        target_transform=transforms.Compose([lambda x: np.array([x]), lambda x: one_hot(x, 10), lambda x: np.squeeze(x)]),
    )

    dataset_test = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size)
    verify_linear_grads()
    verify_sigmoid_grads()
    verify_mlp_grads(train_loader)

    key = random.key(0)
    begin_time = time.time()
    model_params = mlp_init(key, [784, 512, 128, 10])

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss, model_params = train_epoch(args, model_params, train_loader, epoch)
        print(f'\nEpoch {epoch}: Training Loss={train_loss:.3f}')

        acc, test_loss = test_epoch(model_params, test_loader)
        print(f'\nEpoch {epoch}: Test Loss={test_loss:.3f}, Test Accuracy = {acc:.2f}')

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))
        test_accuracies.append(float(acc))

    total_time = time.time() - begin_time
    print(f'time: {total_time:.1f}s')
    epochs = range(1, args.epochs + 1)

    #Loss plot:
    #The training and test loss decrease over the epochs, which shows that the model is learning. 
    #The test loss stays close to the training loss, so there is no strong overfitting.

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy plot:
    #The test accuracy increases quickly in the first epochs and then stabilizes around 98%, showing that the model improves and converges after training.

    plt.figure()
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    learning_rates = [1.0, 0.3, 0.1, 0.03, 0.001]

    all_test_losses = {}
    all_test_accuracies = {}

    begin_time = time.time()

    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}")

        key = random.key(0)
        model_params = mlp_init(key, [784, 512, 128, 10])

        test_losses = []
        test_accuracies = []

        args.lr = lr

        for epoch in range(1, args.epochs + 1):
            train_loss, model_params = train_epoch(args, model_params, train_loader, epoch)

            acc, test_loss = test_epoch(model_params, test_loader)

            test_losses.append(float(test_loss))
            test_accuracies.append(float(acc))

            print(
                f"LR={lr} | Epoch {epoch}: "
                f"Train Loss={train_loss:.3f}, "
                f"Test Loss={test_loss:.3f}, "
                f"Test Accuracy={acc:.2f}"
            )

        all_test_losses[lr] = test_losses
        all_test_accuracies[lr] = test_accuracies

    total_time = time.time() - begin_time
    print(f"\nTotal time: {total_time:.1f}s")

    epochs = range(1, args.epochs + 1)

    plt.figure()
    for lr in learning_rates:
        plt.plot(epochs, all_test_losses[lr], label=f"lr={lr}")

    #Test loss plot:
    #The learning rate strongly affects convergence speed and final loss. Medium learning rates (0.1–0.3) achieve the lowest loss,
    # while a very small learning rate (0.001) learns slowly and a very large one (1.0) is less stable.
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss for Different Learning Rates")
    plt.legend()
    plt.show()

    plt.figure()
    for lr in learning_rates:
        plt.plot(epochs, all_test_accuracies[lr], label=f"lr={lr}")

    #Test accuracy plot:
    #Higher learning rates (0.1–0.3) reach high accuracy quickly, while a very small learning rate (0.001) improves slowly and achieves lower final accuracy.
    #This shows that an appropriate learning rate is crucial for fast and effective training.
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy for Different Learning Rates")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
