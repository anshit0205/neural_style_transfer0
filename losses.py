import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

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

content_weight = 1
style_weights = [0.075, 0.065, 0.085, 0.063, 0.092]
total_variation_weight = 1e-6  # You can adjust this weight

@tf.function
def compute_loss_and_grads(input_image):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        content_output = content_model(input_image)
        style_outputs = style_model(input_image)
        c_loss = content_weight * tf.reduce_mean(tf.square(content_output - content_target))
        s_loss = 0
        for w, symbolic, actual in zip(style_weights, style_outputs, style_layers_outputs):
            s_loss += w * style_loss(symbolic, actual)
        tv_loss = total_variation_weight * total_variation_loss(input_image)
        total_loss = c_loss + s_loss + tv_loss
    grads = tape.gradient(total_loss, input_image)
    return total_loss, grads, c_loss, s_loss, tv_loss

def get_loss_and_grads_wrapper(x_vec):
    x_tensor = tf.convert_to_tensor(x_vec.reshape(*batch_shape), dtype=tf.float32)
    total_loss, grads, content_loss, style_loss, tv_loss = compute_loss_and_grads(x_tensor)
    return total_loss.numpy().astype(np.float64), grads.numpy().flatten().astype(np.float64), content_loss.numpy().astype(np.float64), style_loss.numpy().astype(np.float64), tv_loss.numpy().astype(np.float64)

def minimize_with_lbfgs(fn, epochs, batch_shape):
    x = np.random.randn(np.prod(batch_shape)).astype(np.float32)  # Start with random noise

    total_losses = []
    content_losses = []
    style_losses = []
    tv_losses = []

    for i in range(epochs):
        x, min_val, info = fmin_l_bfgs_b(lambda x: fn(x)[:2], x.flatten(), maxfun=20)
        total_loss, _, content_loss, style_loss, tv_loss = fn(x)
        print(f"Iteration {i}: total_loss={total_loss}, content_loss={content_loss}, style_loss={style_loss}, tv_loss={tv_loss}")

        total_losses.append(total_loss)
        content_losses.append(content_loss)
        style_losses.append(style_loss)
        tv_losses.append(tv_loss)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(content_losses, label='Content Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(style_losses, label='Style Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(tv_losses, label='Total Variation Loss')
    plt.legend()

    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(total_losses, label='Total Loss')
    plt.legend()
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img
