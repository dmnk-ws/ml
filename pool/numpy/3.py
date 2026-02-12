import numpy as np


def f(x):
    return x[0]**2 + x[1]**2


def gradf(x):
    return 2*x[0] + 2*x[1]


if __name__ == "__main__":
    e = 0.1
    x0 = np.array([3, 4], dtype=np.float32)
    N = 10

    x_i = x0
    for i in range(N):
        x_i -= e*gradf(x_i)
        print(x_i)
        print(f(x_i))