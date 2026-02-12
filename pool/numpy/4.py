import numpy as np


def S(x):
    s = np.exp(x)
    s /= np.sum(s, axis=1).reshape(-1, 1)
    return s


if __name__ == "__main__":
    X = np.array([[-1, -1, 5], [1, 1, 2]])
    print(S(X))