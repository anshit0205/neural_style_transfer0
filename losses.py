import tensorflow as tf

def gram_matrix(img):
    img = tf.squeeze(img, axis=0)  # Remove the batch dimension
    X = tf.reshape(tf.transpose(img, (2, 0, 1)), (img.shape[-1], -1))
    G = tf.linalg.einsum('ik,jk->ij', X, X) / tf.cast(tf.size(img), tf.float32)
    return G

def style_loss(y, t):
    return tf.reduce_mean(tf.square(gram_matrix(y) - gram_matrix(t)))

def total_variation_loss(x):
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_mean(a + b)
