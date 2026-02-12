import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(folder: str) -> tuple[np.ndarray, np.ndarray]:
    filenames = os.listdir(folder)
    samples, labels = [], []

    for filename in filenames:
        label = int(filename.split("-")[0])
        labels.append(label)

        filepath = os.path.join(folder, filename)
        img = plt.imread(filepath)
        samples.append(img)

    return np.array(samples), np.array(labels)


def one_hot(labels: np.ndarray) -> np.ndarray:
    labels_copy = labels.copy()

    classes = labels_copy.max() + 1

    return np.eye(classes)[labels_copy]


def remove_white_stripe(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    samples_copy = samples.copy()

    class_indices = labels.argmax(axis=1)
    mask0 = class_indices == 0

    samples_copy[mask0, 3] = samples_copy[mask0, 2]
    samples_copy[mask0] = np.clip(samples_copy[mask0] * 2.0, 0, 1)

    return samples_copy


def oversample_class5(samples: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples_copy = samples.copy()
    labels_copy = labels.copy()

    class_indices = labels.argmax(axis=1)
    mask5 = class_indices == 5
    mask1 = class_indices == 1

    n_duplicates = len(samples_copy[mask1]) // len(samples_copy[mask5]) - 1 # DIRTY

    duplicated_samples = np.tile(samples_copy[mask5], (n_duplicates, 1, 1))
    duplicated_labels = np.tile(labels_copy[mask5], (n_duplicates, 1))

    samples_result = np.concatenate((samples_copy, duplicated_samples))
    labels_result = np.concatenate((labels_copy, duplicated_labels))

    return samples_result, labels_result


def analyse(samples: np.ndarray, labels: np.ndarray) -> None:
    print(f"Samples: {samples.shape}")
    print(f"Labels: {labels.shape}")

    class_indices = labels.argmax(axis=1)

    samples0 = samples[class_indices == 0]
    fig, axs = plt.subplots(1, 1)
    axs.imshow(samples0[0])
    plt.show()

    samples5 = samples[class_indices == 5]
    print(f"Number of samples of class {5}: {len(samples5)}")


if __name__ == '__main__':
    image_dir = sys.argv[1]
    h = int(sys.argv[2])
    w = int(sys.argv[3])
    c = int(sys.argv[4])
    npz = sys.argv[5]
    should_correct = sys.argv[6]

    X, T = load_data(image_dir)
    T = one_hot(T)

    if should_correct == "1":
        analyse(X, T)
        X = remove_white_stripe(X, T)
        X, T = oversample_class5(X, T)
        analyse(X, T)

    X = X.reshape(-1, h, w, c)

    np.savez(npz, X=X, T=T)