import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def show_images(data):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for ax, example in zip(axes.flat, data.take(10)):
        ax.imshow(example['image'])
        ax.set_title(f"{example['label'].numpy()}")

    plt.show()


def preprocess(example):
    image = tf.cast(example['image'], tf.float32) / 255.0
    label = tf.one_hot(example['label'], 10)

    return image, label


def cnn():
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, activation="relu")(inputs)  # 26,26,32
    x = layers.Conv2D(64, 3, activation="relu")(x)  # 24,24,64
    block1output = layers.MaxPooling2D(2)(x)  # 12,12,64

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block1output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block2output = layers.add([x, block1output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block2output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block3output = layers.add([x, block2output])

    x = layers.Conv2D(64, 3, activation="relu")(block3output)

    x = layers.Flatten()(x)
    x = layers.Dense(800, activation="relu")(x)
    output = layers.Dense(10)(x)

    M = tf.keras.Model(inputs, output)

    M.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    M.summary()

    return M


if __name__ == "__main__":
    ds = tfds.load('svhn_cropped', shuffle_files=True)
    train_data = ds['train']
    test_data = ds['test']

    show_images(train_data)

    model = cnn()

    train_data = train_data.map(preprocess).batch(100)
    test_data = test_data.map(preprocess).batch(100)

    model.fit(train_data, epochs=3)
    model.evaluate(test_data)