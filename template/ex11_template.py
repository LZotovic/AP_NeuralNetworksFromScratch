import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Functions to be completed
# -------------------------------------------------------
def ground_truth_function(x: np.ndarray) -> np.ndarray:
    """
    TODO:
    Replace this placeholder with the ground truth function used in the lecture:
        h(x) = sin(2πx)
    """
    return np.sin(2 * np.pi * x)  # placeholder


def error_function(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    TODO:
    Replace this placeholder with the non-regularized error function from the lecture.
    Hint: start from the sum of squared errors and, later on, compute the RMS error.
    """
    return float(0.5 * np.sum((y_pred - y_true) ** 2))  # placeholder

def rms_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    e = error_function(y_pred, y_true)
    n = len(y_true)
    return float(np.sqrt((2 * e) / n))


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
        plt.plot(xs, model(xs), label=f'Model (deg={getattr(model, "degree", lambda: "?")()})')
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
    x_test  = np.linspace(0.0, 1.0, n_samples)
    y_train = ground_truth_function(x_train) + rng.normal(0.0, noise_amplitude, size=n_samples)
    y_test  = ground_truth_function(x_test)  + rng.normal(0.0, noise_amplitude, size=n_samples)

    # Plot 1 – initial data (the ground truth is wrong until the TODOs are solved)
    plot_model(x_train, y_train, x_test, y_test, model=None, save_fname='Initial_data.pdf')

    # Fit a polynomial of degree 3
    degree = 3
    model_deg3 = np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)

    # Compute and display the current training and test error
    y_pred_train = model_deg3(x_train)
    y_pred_test  = model_deg3(x_test)
    train_err = error_function(y_pred_train, y_train)
    test_err  = error_function(y_pred_test,  y_test)
    print(f"[deg={degree}] train_err={train_err:.6f}, test_err={test_err:.6f}")

    # Plot 2 – polynomial fit of degree 3
    plot_model(x_train, y_train, x_test, y_test, model=model_deg3, save_fname='Initial_fit.pdf')

    # -------------------------------------------------------
    # Continue here once the initial plots look correct:
    # -------------------------------------------------------

    # 1) Fit and plot an overfitted polynomial of degree 11. 
    overfit_degree = 11
    model_deg11 = np.polynomial.Polynomial.fit(x_train, y_train, deg=overfit_degree)

    y_pred_train_11 = model_deg11(x_train)
    y_pred_test_11 = model_deg11(x_test)

    train_rms_11 = rms_error(y_pred_train_11, y_train)
    test_rms_11 = rms_error(y_pred_test_11, y_test)

    print(f"[deg={overfit_degree}] train_rms={train_rms_11:.6f}, test_rms={test_rms_11:.6f}")

    plot_model(x_train, y_train, x_test, y_test, model=model_deg11, save_fname='Overfitted_deg11.pdf')
    # 2) Vary the polynomial degree from 0 to 11.
    #    Compute the RMS training and test errors and reproduce
    #    the plot “Polynomial degree vs. train/test error”.    
    degrees = list(range(12))
    train_rms_list = []
    test_rms_list = []

    for deg in degrees:
        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=deg)

        y_pred_train = model(x_train)
        y_pred_test = model(x_test)

        train_rms_list.append(rms_error(y_pred_train, y_train))
        test_rms_list.append(rms_error(y_pred_test, y_test))

    plt.figure()
    plt.plot(degrees, train_rms_list, 'ob-', label='Training')
    plt.plot(degrees, test_rms_list, 'or-', label='Test')
    plt.xlabel('Polynomial degree')
    plt.ylabel('RMS error')
    plt.xticks(degrees)
    plt.legend()
    plt.savefig('Degree_vs_RMS.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    # 3) Keep the polynomial degree fixed at 10.
    #    Vary the dataset size until the function does not overfit anymore.
    #    Plot how train and test RMS errors change with data size.
    fixed_degree = 10
    sample_sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
    train_rms_by_size = []
    test_rms_by_size = []

    for n_samples_current in sample_sizes:
        x_train_cur = np.linspace(0.0, 1.0, n_samples_current)
        x_test_cur  = np.linspace(0.0, 1.0, n_samples_current)

        y_train_cur = ground_truth_function(x_train_cur) + rng.normal(
            0.0, noise_amplitude, size=n_samples_current
        )
        y_test_cur = ground_truth_function(x_test_cur) + rng.normal(
            0.0, noise_amplitude, size=n_samples_current
        )

        model = np.polynomial.Polynomial.fit(x_train_cur, y_train_cur, deg=fixed_degree)

        y_pred_train_cur = model(x_train_cur)
        y_pred_test_cur = model(x_test_cur)

        train_rms_by_size.append(rms_error(y_pred_train_cur, y_train_cur))
        test_rms_by_size.append(rms_error(y_pred_test_cur, y_test_cur))

    plt.figure()
    plt.plot(sample_sizes, train_rms_by_size, 'ob-', label='Training')
    plt.plot(sample_sizes, test_rms_by_size, 'or-', label='Test')
    plt.xlabel('Number of samples')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('Dataset_size_vs_RMS.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()