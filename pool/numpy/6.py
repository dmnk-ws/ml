import numpy as np


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    print(relu(np.array([[-1, 1], [1, -5]])))
    print(relu(np.array([[-1, 1, 0], [1, 2, -5]])))