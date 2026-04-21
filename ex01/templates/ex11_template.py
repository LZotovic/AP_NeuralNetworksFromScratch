import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Functions to be completed
# -------------------------------------------------------


# Ground truth function h(x) = sin(2πx) from the lecture
# Used to generate the data(with added noise)
def ground_truth_function(x: np.ndarray) -> np.ndarray:
    return np.sin(2*np.pi*x)


# Function that computes the RMS (Root Mean Square) error
# Measures the average difference between predicted (y_pred) and true (y_true) values.
def error_function(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    N = len(y_pred)
    s = 0.0
    for i in range(N):
        s += (y_pred[i] - y_true[i]) ** 2
    return 0.5 * s


# -------------------------------------------------------
# Helper for visualization
# -------------------------------------------------------
def plot_model(x_train, y_train, x_test, y_test, model=None, save_fname=None):
    xs = np.linspace(0.0, 1.0, 1000)
    plt.figure()
    plt.plot(xs, ground_truth_function(xs), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test,  y_test,  'xr', label='Test data')
    if model is not None:
        plt.plot(
            xs, model(xs), label=f'Model (deg={getattr(model, "degree", lambda: "?")()})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if save_fname is not None:
        plt.savefig(save_fname, bbox_inches='tight')
    plt.show()
    plt.clf()


# -------------------------------------------------------
# Main procedure
# -------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    n_samples = 11
    noise_amplitude = 0.15

    # Generate noisy training and test data
    x_train = np.linspace(0.0, 1.0, n_samples)
    x_test = np.linspace(0.0, 1.0, n_samples)
    y_train = ground_truth_function(
        x_train) + rng.normal(0.0, noise_amplitude, size=n_samples)
    y_test = ground_truth_function(
        x_test) + rng.normal(0.0, noise_amplitude, size=n_samples)

    # Plot 1 – initial data
    # Expected:
    # The training data and test data should be close to the sine curve, since they are generated
    # from the ground truth function with added noise.
    # Observed:
    # Both the training points and test points are near the sine curve
    # The overall shape is visible, with small deviations due to the noise
    plot_model(x_train, y_train, x_test, y_test,
               model=None, save_fname='Initial_data.pdf')

    # Fit a polynomial of degree 3
    degree = 3
    model_deg3 = np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)

    # Compute and display the current training and test error
    y_pred_train = model_deg3(x_train)
    y_pred_test = model_deg3(x_test)
    train_err = error_function(y_pred_train, y_train)
    test_err = error_function(y_pred_test,  y_test)
    print(f"[deg={degree}] train_err={train_err:.6f}, test_err={test_err:.6f}")

    # Plot 2 – polynomial fit of degree 3
    # Expected:
    # Model should follow the general shape of the sine curve, without going through every training point
    # Observed:
    # The model follows the curve quite well, it doesn't pass through all training points and
    # it's not overfitting, because it stays close to both training and test points
    plot_model(x_train, y_train, x_test, y_test,
               model=model_deg3, save_fname='Initial_fit.pdf')

    # -------------------------------------------------------
    # Continue here once the initial plots look correct:
    # -------------------------------------------------------

    # 1) Fit and plot an overfitted polynomial of degree 11.
    degree = 11
    model_deg11 = np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)
    # Plot 3 - polynomial fit of degree 11
    # Expected:
    # Model would try to fit almost every training point, leading to a very wiggly curve
    # Observed:
    # Model is overfitting and wavy because it passes through the training points
    # But it doesn't match the test points well
    plot_model(x_train, y_train, x_test, y_test,
               model=model_deg11, save_fname='Polynomial_fit_deg11.pdf')

    # 2) Vary the polynomial degree from 0 to 11.
    #    Compute the RMS training and test errors and reproduce
    #    the plot “Polynomial degree vs. train/test error”.
    #
    def rms_error_function(y_pred, y_true):
        return np.sqrt(2 * error_function(y_pred, y_true) / len(y_pred))

    train_errors = []
    test_errors = []
    degrees = range(0, 12)
    for degree in degrees:
        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)

        y_pred_train = model(x_train)
        y_pred_test = model(x_test)
        train_err = rms_error_function(y_pred_train, y_train)
        test_err = rms_error_function(y_pred_test,  y_test)
        train_errors.append(train_err)
        test_errors.append(test_err)
        print(
            f"[deg={degree}] train_err={train_err:.6f}, test_err={test_err:.6f}")
    # Plot 3 - Polynomial degree vs. train/test error
    # Expected:
    # Increasing the polynomial degree should lower the training error,
    # as the model can fit the training data better
    # A decrease in the testing error is expected, then with increase in polynomial degree
    # it should increase again when the model starts overfitting
    # Observed:
    # Higher degree causes the training error to decrease
    # It can be seen that small degrees underfit, and large degrees overfit, that is why
    # firstly the test error decreases, but then increases again
    plt.plot(degrees, train_errors, label="Training error")
    plt.plot(degrees, test_errors, label="Testing error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Train/Test error")
    plt.legend()
    plt.grid()
    plt.savefig("Polynomial_degree_vs_error.pdf")
    plt.show()

    # 3) Keep the polynomial degree fixed at 10.
    #    Vary the dataset size until the function does not overfit anymore.
    #    Plot how train and test RMS errors change with data size.

    # Vary data size
    rng = np.random.default_rng(42)
    n_samples = [10, 20, 50, 100, 200, 500, 700, 1000, 1500]
    noise_amplitude = 0.15
    degree = 10
    train_errors = []
    test_errors = []

    for sample in n_samples:
        x_train = np.linspace(0.0, 1.0, sample)
        x_test = np.linspace(0.0, 1.0, sample)
        y_train = ground_truth_function(
            x_train) + rng.normal(0.0, noise_amplitude, size=sample)
        y_test = ground_truth_function(
            x_test) + rng.normal(0.0, noise_amplitude, size=sample)

        model_deg10 = np.polynomial.Polynomial.fit(
            x_train, y_train, deg=degree)

        y_pred_train = model_deg10(x_train)
        y_pred_test = model_deg10(x_test)
        train_err = rms_error_function(y_pred_train, y_train)
        test_err = rms_error_function(y_pred_test,  y_test)

        train_errors.append(train_err)
        test_errors.append(test_err)

        print(
            f"[deg={degree}] train_err={train_err:.6f}, test_err={test_err:.6f}")

    # Plot 4 - Change of train and test error with the number of samples
    # Expected:
    # Model should overfit for small number of samples, the training error should be low and the test error higher
    # Increasing the number of samples should make the model generalize better,
    # and we expect a smaller difference between training and test error
    # Observed:
    # Overfitting can be observed for small datasets, as the training error is low while the test error is higher
    # Both errors converge to a similar value as the number of samples increases => model generalizes better and overfits less
    plt.plot(n_samples, train_errors, label="Training error")
    plt.plot(n_samples, test_errors, label="Testing error")
    plt.xlabel("Number of samples")
    plt.ylabel("RMS error")
    plt.legend()
    plt.grid()
    plt.savefig("Samples_vs_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
