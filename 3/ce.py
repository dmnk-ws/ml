import numpy as np

def ce(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return -(B*np.log(A)).sum(axis=1).mean()

def ce2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    print(range(0, A.shape[0]))
    print(np.argmax(B, axis=1))
    print(A[range(0, A.shape[0]), np.argmax(B, axis=1)])
    return -np.log(Y[range(0, Y.shape[0]), np.argmax(T, axis=1)]).mean()

if __name__ == "__main__":
    Y = np.array([0, 0, 0, 0.5, 0, 0.3, 0, 0.2, 0, 0])
    T = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    #Y = np.array([[0.4, 0.4, 0.2], [0.7, 0.1, 0.2]])
    #T = np.array([[1, 0, 0], [0, 0., 1]])
    print(ce(Y, T))