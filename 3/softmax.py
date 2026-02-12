import numpy as np

def softmax(A: np.ndarray) -> np.ndarray:
    B = A.copy()
    B = np.exp(B)
    B /= np.sum(B, axis=1, keepdims=True)
    return B

if __name__ == "__main__":
    X = np.array([[-1, -1, 5], [1, 1, 2]])
    print(softmax(X))