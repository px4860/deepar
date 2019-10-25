import tensorflow as tf

def gaussian_likelihood(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.math.abs(tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma))) + 1e-6
    return gaussian_loss


def gaussian_likelihood_2(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(tf.compat.v2.keras.backend.abs(tf.contrib.distributions.percentile(
            tf.compat.v2.keras.backend.random_normal(shape=(20,15,100), mean=y_pred, stddev=sigma), 50, axis=1) - y_true)) + 1e-6
    return gaussian_loss