import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


WEIGHTS_FILE = "cnn.weights.h5"


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_dir)
    samples = data['X']
    labels = data['T']

    return samples, labels


def build_cnn(input_shape: tuple[int, ...]) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, (5, 5)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (5, 5)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Reshape(target_shape=(4 * 4 * 32,)),

        tf.keras.layers.Dense(100),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


def train(samples: np.ndarray, labels: np.ndarray, cnn: tf.keras.Sequential) -> None:
    N = samples.shape[0]
    rng = np.random.default_rng(42)
    indices = rng.permutation(N)

    split = int(0.8 * N)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, T_train = samples[train_idx], labels[train_idx]
    X_test, T_test = samples[test_idx], labels[test_idx]

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    cnn.fit(X_train, T_train, epochs=5, batch_size=100)

    acc = cnn.evaluate(X_test, T_test)
    print(f"CNN Train Accuracy: {acc[1]:.4f}")

    cnn.save_weights(WEIGHTS_FILE)


def test(samples: np.ndarray, labels: np.ndarray, cnn: tf.keras.Sequential) -> None:
    cnn.load_weights(WEIGHTS_FILE)
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    acc = cnn.evaluate(samples, labels)
    print(f"CNN Test Accuracy: {acc[1]:.4f}")

    predictions = cnn.predict(samples)

    Y_true = np.argmax(labels, axis=1)
    Y_pred = np.argmax(predictions, axis=1)

    cm = tf.math.confusion_matrix(Y_true, Y_pred).numpy()
    plt.imshow(cm)
    plt.show()


if __name__ == '__main__':
    npz = sys.argv[1]
    mode = sys.argv[2]

    X, T = load_data(npz)
    model = build_cnn(X.shape[1:])

    if mode == "train":
        train(X, T, model)
    else:
        test(X, T, model)