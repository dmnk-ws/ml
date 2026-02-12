import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    if not os.path.exists("./mnist.npz"):
        print ("Downloading MNIST...")
        fname = 'mnist.npz'
        url = 'http://www.gepperth.net/alexander/downloads/'
        r = requests.get(url+fname)
        open(fname , 'wb').write(r.content)

    ## read it into
    data = np.load("mnist.npz")
    traind = data["arr_0"]
    trainl = data["arr_2"]
    traind = traind.reshape(60000,28,28)

    if sys.argv[1] == "1a":
        print(np.arange(0, 101, 2))

    if sys.argv[1] == "1b":
        print(np.ones((3, 3)) * np.array([1, 2, 3]).reshape(-1, 1))

    if sys.argv[1] == "1c":
        print(np.ones((3, 5)) * 55)

    if sys.argv[1] == "1d":
        print(np.random.uniform(0, 1, (5, 4, 3)))

    if sys.argv[1] == "2a":
        sample1000 = traind[999] + 0.0

        fig, ax = plt.subplots(1, 1)
        ax.imshow(sample1000)
        plt.show()

    if sys.argv[1] == "2b":
        sample1000 = traind[999] + 0.0
        sample1000[:, :10] = 0
        sample1000[:, -10:] = 0

        fig, ax = plt.subplots(1, 1)
        ax.imshow(sample1000)
        plt.show()

    if sys.argv[1] == "2c":
        sample10 = traind[9] + 0.0

        v1 = sample10[::2]
        v2 = sample10[:,::2]
        v3 = sample10[::-1, ::-1]
        v4 = sample10[::-2, ::-2]

        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(v1)
        ax[0,1].imshow(v2)
        ax[1,0].imshow(v3)
        ax[1,1].imshow(v4)
        plt.show()

    if sys.argv[1] == "2d":
        copy = traind + 0.0
        #copy *= -1
        #copy += 1
        i = 1 - copy

        fig, ax = plt.subplots(1, 1)
        ax.imshow(i[1])
        plt.show()

    if sys.argv[1] == "3a":
        x = traind + 0.0
        x += np.arange(1, 29).reshape(-1, 1)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(x[99])
        plt.show()

    if sys.argv[1] == "3b":
        x = traind + 0.0
        x += np.arange(1, 29).reshape(1, -1)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(x[99])
        plt.show()

    if sys.argv[1] == "4a":
        vec20 = np.arange(1, 21)
        mask_idx = vec20 < 10
        less10 = vec20[mask_idx]
        print(less10)

    if sys.argv[1] == "4b":
        vec20 = np.arange(1, 21)
        fancy_idx = [1, 5, 19]
        vec20[fancy_idx] = 0
        print(vec20)

    if sys.argv[1] == "4c":
        copy = traind + 0.0
        samples = copy[[1, 5, 19]]
        print(samples.shape)

    if sys.argv[1] == "5a":
        copy = traind + 0.0
        rnd_samples = copy[np.random.choice(60000, 10)]
        #rnd_samples = copy[np.random.randint(0, 60000, size=[10])]

        fig, ax = plt.subplots(2, 5, figsize=(12, 5))
        for i, a in enumerate(ax.flat):
            a.imshow(rnd_samples[i])
        plt.show()

    if sys.argv[1] == "5b":
        copy = traind + 0.0
        mask = trainl.argmax(axis=1) == 0
        samples0 = copy[mask]
        print(samples0.shape)