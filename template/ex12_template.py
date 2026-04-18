import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------
# Data generation, same as in exercise 1.1
# ---------------------------------------------------------
def ground_truth_function(x: np.ndarray) -> np.ndarray:
    """
    TODO:
    Replace this placeholder with the ground truth function used in the lecture:
        h(x) = sin(2πx)
    """
    return np.sin(2 * np.pi * x)  # placeholder


# ---------------------------------------------------------
# Model and loss functions
# ---------------------------------------------------------
def poly_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    TODO:
    Implement the polynomial model, returning predictions for input x and weights w.
    returns: shape (N,)
    """
    y_pred = np.zeros_like(x, dtype=float)
    for i in range(len(w)):
        y_pred += w[i] * (x ** i)

    return y_pred


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    TODO:
    Implement the Mean Squared Error (MSE) loss function:
    """
    return float(0.5 * np.mean((y_pred - y_true) ** 2))


# ---------------------------------------------------------
# SGD update step
# ---------------------------------------------------------
def sgd_update(x_b: np.ndarray, y_b: np.ndarray, w: np.ndarray, eta: float) -> np.ndarray:
    """
    TODO:
    Implement the SGD update for a batch (full-batch if x_b is the full dataset):
      1) compute y_pred
      2) compute gradient dL/dw_j for each j
      3) return updated weights: w_new = w - eta * dL/dw (where dL/dw is the gradient vector)
    """
    y_pred = poly_predict(x_b, w)
    batch_size = len(x_b)

    grad = np.zeros_like(w, dtype=float)

    for j in range(len(w)):
        grad[j] = np.sum((y_pred - y_b) * (x_b ** j)) / batch_size

    w_new = w - eta * grad
    return w_new


# ---------------------------------------------------------
# Training loops
# ---------------------------------------------------------
def train_fullbatch(x: np.ndarray, y: np.ndarray, degree: int, eta: float, n_epochs: int, seed: int = 0):
    """
    TODO:
    1) initialize w ~ N(0, 0.1)
    2) for epoch in 1..n_epochs:
         w = sgd_update(x, y, w, eta)
         record training loss on full x,y
    return (w, losses)
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=degree + 1)
    losses = []

    for epoch in range(n_epochs):
        w = sgd_update(x, y, w, eta)

        y_pred = poly_predict(x, w)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

    return w, losses


def train_minibatch(x: np.ndarray, y: np.ndarray, degree: int, eta: float, batch_size: int, n_epochs: int, seed: int = 0):
    """
    TODO:
    1) initialize w ~ N(0, 0.1)
    2) for epoch in 1..n_epochs:
         - shuffle (x,y)
         - for each mini-batch of size batch_size:
             w = sgd_update(x_batch, y_batch, w, eta)
         - record training loss on full x,y
    return (w, losses)
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=degree + 1)
    losses = []
    n = len(x)

    for epoch in range(n_epochs):
        indices = rng.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for start in range(0, n, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            w = sgd_update(x_batch, y_batch, w, eta)

        y_pred = poly_predict(x, w)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

    return w, losses

def train_minibatch_snapshots(x: np.ndarray, y: np.ndarray, degree: int, eta: float, batch_size: int, n_epochs: int, seed: int = 0, snapshot_every: int = 10):
    """
    Similar to train_minibatch, but also collects and returns snapshots of w every 'snapshot_every' epochs.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=degree + 1)
    losses = []
    snapshots = []
    n = len(x)

    for epoch in range(n_epochs):
        indices = rng.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for start in range(0, n, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            w = sgd_update(x_batch, y_batch, w, eta)

        y_pred = poly_predict(x, w)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

        if epoch % snapshot_every == 0:
            snapshots.append(w.copy())

    return w, losses, snapshots


# ---------------------------------------------------------
# Example animation helper (adapt this as needed)
# ---------------------------------------------------------
def create_gif(x: np.ndarray, y: np.ndarray, snapshots: list[np.ndarray], snapshot_every: int, path: str = "SGD_training_animation.gif"):
    """
    Example helper for visualization. You may adjust styling/axes if desired.
    To produce snapshots, collect w every 'snapshot_every' epochs during training.
    """
    xs = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots()
    (line_model,) = ax.plot([], [], lw=2, label="Model prediction")
    ax.plot(x, y, 'o', label="Training data")
    ax.plot(xs, ground_truth_function(xs), 'k--', label="Ground truth")
    ax.set_xlim(0, 1); ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.set_title("Polynomial learning dynamics (SGD)")

    def init():
        line_model.set_data([], [])
        return (line_model,)

    def update(frame):
        w = snapshots[frame]
        y_pred = poly_predict(xs, w)
        line_model.set_data(xs, y_pred)
        ax.set_title(f"Epoch {frame * snapshot_every}")
        return (line_model,)

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), init_func=init, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=10))
    plt.close()


# ---------------------------------------------------------
# Main (explicit, minimal)
# ---------------------------------------------------------
def main():
    # Data (fixed N=200)
    rng = np.random.default_rng(42)
    N = 200
    noise = 0.15
    x = np.linspace(0.0, 1.0, N)
    y = ground_truth_function(x) + rng.normal(0.0, noise, size=N)

    # Choose degree (from Part 1)
    degree = 5  # TODO: adjust based on your Part 1 result

    # ---------- 1) Full-batch training ----------
    # TODO: uncomment, choose a learning rate and train for 2000 epochs, then plot/save the loss curve
    eta_fb = 0.03  # this one will likely be too high, adjust as needed
    w_fb, losses_fb = train_fullbatch(x, y, degree, eta=eta_fb, n_epochs=2000)
    plt.figure() 
    plt.plot(losses_fb, label=f"full-batch, eta={eta_fb}")
    plt.xlabel("Epoch") 
    plt.ylabel("Training Loss (MSE)")
    plt.legend()
    plt.savefig("SGD_fullbatch_loss.pdf", bbox_inches="tight")
    plt.close()

    # ---------- 2) Learning-rate sweep (fixed batch=16) ----------
    # TODO: uncomment and select a few learning rates in [1e-4, 1] for 2000 epochs
    etas = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1] # these may be too extreme, adjust as needed
    plt.figure()
    for eta in etas:
        _, losses = train_minibatch(x, y, degree, eta=eta, batch_size=16, n_epochs=2000)
        plt.plot(losses, label=f"lr={eta}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.legend()
    plt.title("Effect of learning rate on convergence (b=16)")
    plt.savefig("SGD_learning_rate_comparison.pdf", bbox_inches="tight")
    plt.close()

    # ---------- 3) Batch-size sweep (fixed learning rate) ----------
    # TODO: uncomment and select a few batch sizes in [1, N] for 2000 epochs
    bs_list = [1, 4, 16, 64, N]
    eta_fixed = 0.03 # this one will likely be too high, adjust as needed
    plt.figure()
    for b in bs_list:
        _, losses = train_minibatch(x, y, degree, eta=eta_fixed, batch_size=b, n_epochs=2000)
        plt.plot(losses, label=f"b={b}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.legend()
    plt.title(f"Effect of batch size on convergence (lr={eta_fixed})")
    plt.savefig("SGD_batch_size_comparison.pdf", bbox_inches="tight")
    plt.close()

    # ---------- 4) Animation ----------
    # TODO: uncomment, choose good (eta, b), collect snapshots during training, and create a GIF
    eta_anim, b_anim = 0.03, 16
    epochs_anim, snap_every = 2000, 1
    _, _, snapshots = train_minibatch_snapshots(x, y, degree, eta=eta_anim, batch_size=b_anim, n_epochs=epochs_anim, snapshot_every=snap_every)
    create_gif(x, y, snapshots, snapshot_every=snap_every, path="SGD_training_animation.gif")


if __name__ == "__main__":
    main()