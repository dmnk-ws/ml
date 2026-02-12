import tensorflow as tf

def softmax(x: tf.Tensor) -> tf.Tensor:
    exp = tf.exp(x)
    return exp / tf.reduce_sum(exp, axis=-1, keepdims=True) # axis=-1 takes the innermost/last axis

def relu(x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(x, 0)

if __name__ == "__main__":
    x1 = tf.constant([[1, 2, -3], [0, 0, 1]], dtype=tf.float32)
    x2 = tf.constant([1, -2], dtype=tf.float32)

    print("Input tensor x1:")
    print(x1)
    print("Softmax x1:")
    print(softmax(x1))
    print("Relu x1:")
    print(relu(x1), end="\n\n")

    print("Input tensor x2:")
    print(x2)
    print("Softmax x2:")
    print(softmax(x2))
    print("Relu x2:")
    print(relu(x2))