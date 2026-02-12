import numpy as np


def CE(y, t):
    log_y = np.log(y)
    indices_t = np.argmax(t, axis=1)
    log_y_i = log_y[:, indices_t]
    return -np.mean(log_y_i)


if __name__ == "__main__":
    #pred0 = np.array([0, 0, 0, 0.5, 0, 0.3, 0, 0.2, 0, 0])
    #label0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    pred0 = np.array([[0.4, 0.4, 0.2], [0.7, 0.1, 0.2]])
    label0 = np.array([[1, 0, 0], [0, 0., 1]])

    print(CE(pred0, label0))

