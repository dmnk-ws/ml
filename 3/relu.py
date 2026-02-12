import numpy as np

def relu(A: np.ndarray) -> np.ndarray:
    B = A.copy()
    B[B < 0] = 0
    return B

if __name__ == "__main__":
    X = np.array([[-1, -1, 5], [1, 1, 2]])
    print(relu(X))