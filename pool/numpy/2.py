import numpy as np
import sys


if __name__ == "__main__":
    task = sys.argv[1]
    td = np.ones((50, 5, 5)) * np.arange(50).reshape(-1, 1, 1)

    if task == "a":
        x = td[0]
        print(x)

    if task == "b":
        sample10 = td[9]
        sample10[:,-2:] = -1
        print(sample10)

    if task == "c":
        sample10 = td[9]
        print(np.mean(sample10))

    if task == "d":
        sample10 = td[9]

        if sys.argv[2] == "1":
            z = sample10[::3]
            print(z)

        if sys.argv[2] == "2":
            z = sample10[:, ::3]
            print(z)

        if sys.argv[2] == "3":
            #sample10 = np.ones((5, 5)) * np.arange(25).reshape(5, 5)
            sample10 = td[9]
            z = sample10[::-1]
            print(z)

        if sys.argv[2] == "4":
            #sample10 = np.ones((5, 5)) * np.arange(25).reshape(5, 5)
            sample10 = td[9]
            z = sample10[::-2]
            print(z)

    if task == "e":
        td += 1 # in-place!!
        print(td)

    if task == "f":
        f = np.array(td[[4,6,9]])
        print(f)

    if task == "g":
        m = np.array(td[np.mean(td, axis=(1,2)) > 45])
        print(m)
