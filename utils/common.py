import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if tf.cuda.is_available():
        tf.cuda.manual_seed(seed)
        tf.cuda.manual_seed_all(seed)
        tf.backends.cudnn.deterministic = True
        tf.backends.cudnn.benchmark = False
    else:
        tf.manual_seed(seed)


def get_files(directory, extension):
    for root, _, files in os.walk(directory):
        return [os.path.join(root, file) for file in files
                if os.path.splitext(file)[1] == extension]


def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.001)
