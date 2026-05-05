import numpy as np
import matplotlib.pyplot as plt

gpu = np.load("gpu_results.npz")
cpu = np.load("cpu_results.npz")

plt.figure()
plt.plot(gpu["times"], gpu["test_accuracies"], label="GPU")
plt.plot(cpu["times"], cpu["test_accuracies"], label="CPU")

plt.xlabel("Time (seconds)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy over Time: CPU vs GPU")
plt.legend()
plt.grid(True)

plt.savefig("cpu_vs_gpu_accuracy_time.pdf")
plt.close()



