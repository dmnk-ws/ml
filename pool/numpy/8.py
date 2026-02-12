import numpy as np
import sys


if __name__ == "__main__":
    task = sys.argv[1]

    if task == "a":
        y = np.random.randint(0, 4, 20)
        t = np.random.randint(0, 4, 20)
        num_classes = len(np.unique(t))

        confusion1 = np.zeros((num_classes, num_classes))
        for true, pred in zip(t, y):
            confusion1[true, pred] += 1
        print(confusion1)

        confusion2 = np.zeros((num_classes, num_classes))
        np.add.at(confusion2, (t, y), 1)
        print(confusion2)

    if task == "b":
        labels = np.random.randint(0, 10, 100)
        one_hot = np.eye(10)[labels]
        print(labels[0])
        print(one_hot[0])

    if task == "c":
        traind = np.ones((20, 10)) * np.arange(0, 20).reshape(-1, 1)
        trainl = np.eye(20)

        rnd_indices = np.random.permutation(len(traind))
        traind = traind[rnd_indices]
        trainl = trainl[rnd_indices]
        print(traind)
        print(trainl)