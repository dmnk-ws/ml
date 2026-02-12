import numpy as np
import sys

if __name__ == "__main__":
    task = sys.argv[1]

    if task == "1a":
        print(np.arange(0, 100, 2))

    if task == "1b":
        print(np.ones((100, 3)) * np.arange(1, 101).reshape(-1, 1))

    if task == "1c":
        print(np.ones((3, 5)) * 55)

    if task == "1d":
        print(np.random.uniform(0, 1, (5, 4, 3)))