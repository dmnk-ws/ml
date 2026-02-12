import numpy as np
import matplotlib.pyplot as plt

def grad(X: np.ndarray) -> np.ndarray | None:
    if X.shape[0] != 2:
        return None
    return np.array([2, 4]) * X

def grad_descent(X: np.ndarray, n: int, e: float = 0.001) -> np.ndarray:
    trajectory = [X]
    X_old = X + 0.0
    for i in range(n):
        X_new = X_old - e * grad(X_old)
        trajectory.append(X_new)
        X_old = X_new + 0.0
        print(X_new)
    return np.array(trajectory)

def plot_trajectory(trajectory: np.ndarray) -> None:
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o')
    plt.gca().invert_xaxis()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Gradient Descent')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    eps = 0.1
    X_i = np.array([1, 3])
    N = 10

    plot_trajectory(
        grad_descent(X_i, N, eps)
    )

