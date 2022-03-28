from tensorflow import keras


def hard_swish(x):
    return x * (keras.activations.relu(x + 3., max_value = 6.) / 6.)