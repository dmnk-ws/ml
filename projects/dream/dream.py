import sys
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def load_image(path: str):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = tf.expand_dims(img, axis=0)

    # return img
    return tf.Variable(img)


if __name__ == "__main__":
    image_path = sys.argv[1]
    X = int(sys.argv[2])
    N = int(sys.argv[3])

    image = load_image(image_path)

    model = tf.keras.applications.MobileNetV2()

    dream_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(index=X).output
    )

    for i in range(N):
        with tf.GradientTape() as tape:
            # tape.watch(image)
            output = dream_model(image)
            loss = tf.reduce_sum(output**2)

            gradient = tape.gradient(loss, image)
            gradient /= tf.math.reduce_mean(tf.math.abs(gradient))
            # image = image + 0.001*gradient
            image.assign_add(0.01*gradient)

        if i % 10 == 0:
            print(i, loss.numpy())
            plt.imshow(np.clip(tf.constant(image).numpy()[0], 0, 1))
            plt.savefig(f"output/dream_{i}_{image_path}")