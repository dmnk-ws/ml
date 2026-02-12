import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    x = np.linspace(1, 5, 100)
    y = 1/x

    img = np.random.uniform(0, 1, (20, 20))

    ax[0].plot(x, y)
    ax[1].scatter(x, y)
    ax[2].bar(x, y)
    ax[3].imshow(img)
    plt.show()
