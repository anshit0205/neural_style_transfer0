import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import lpips
import torch
import tensorflow_gan as tfgan
from skimage.metrics import structural_similarity as ssim

from utils import psnr, load_img_and_preprocess, unpreprocess, scale_img
from losses import style_loss, total_variation_loss
from models import VGG16_Avgpool, get_content_model, get_style_model

content_weight = 1
style_weights = [0.075, 0.065, 0.085, 0.063, 0.092]
total_variation_weight = 1e-6

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
    x = np.random.randn(np.prod(batch_shape)).astype(np.float32) # Start from the content image instead of random noise

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

def display_images(content_img_np, style_img_np, final_img_np):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(scale_img(content_img_np))
    plt.title('Content Image')

    plt.subplot(1, 2, 2)
    plt.imshow(scale_img(style_img_np))
    plt.title('Style Image')

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(scale_img(final_img_np))
    plt.title('Result Image')
    plt.show()

def display_histograms(content_img_np, style_img_np, final_img_np):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(content_img_np.ravel(), bins=256, color='red', alpha=0.5)
    plt.title('Content Image Histogram')

    plt.subplot(1, 3, 2)
    plt.hist(style_img_np.ravel(), bins=256, color='blue', alpha=0.5)
    plt.title('Style Image Histogram')

    plt.subplot(1, 3, 3)
    plt.hist(final_img_np.ravel(), bins=256, color='green', alpha=0.5)
    plt.title('Result Image Histogram')

    plt.show()

if __name__ == '__main__':
    content_img_path = 'path_to_content_image.png'
    style_img_path = 'path_to_style_image.png'

    content_img = load_img_and_preprocess(content_img_path)
    h, w = content_img.shape[1:3]
    style_img = load_img_and_preprocess(style_img_path, (h, w))

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    content_layer_name = 'block4_conv2'
    style_layer_names = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv1',
        'block5_conv2'
    ]

    vgg = VGG16_Avgpool(shape)
    content_model = get_content_model(vgg, content_layer_name)
    style_model = get_style_model(vgg, style_layer_names)

    content_target = tf.constant(content_model.predict(content_img))
    style_layers_outputs = [tf.constant(output) for output in style_model.predict(style_img)]

    final_img = minimize_with_lbfgs(get_loss_and_grads_wrapper, 11, batch_shape)

    content_img_np = unpreprocess(content_img[0])
    style_img_np = unpreprocess(style_img[0])

    display_images(content_img_np, style_img_np, final_img[0])

    # SSIM
    content_gray = tf.image.rgb_to_grayscale(content_img_np)
    result_gray = tf.image.rgb_to_grayscale(final_img[0])
    ssim_index = ssim(content_gray.numpy().squeeze(), result_gray.numpy().squeeze(), data_range=result_gray.numpy().max() - result_gray.numpy().min())
    print(f'SSIM: {ssim_index}')

    # PSNR
    psnr_value = psnr(content_img_np, final_img[0])
    print(f'PSNR: {psnr_value} dB')

    display_histograms(content_img_np, style_img_np, final_img[0])

    # LPIPS
    lpips_fn = lpips.LPIPS(net='alex')

    content_img_tensor = torch.tensor(content_img_np).permute(2, 0, 1).unsqueeze(0)
    final_img_tensor = torch.tensor(final_img[0].copy().transpose(2, 0, 1)).unsqueeze(0)

    content_img_tensor = content_img_tensor.float()
    final_img_tensor = final_img_tensor.float()

    lpips_score = lpips_fn(content_img_tensor, final_img_tensor)
    print(f'LPIPS: {lpips_score.item()}')

    # KID
    content_img_resized = tf.image.resize(content_img_np, [299, 299])
    final_img_resized = tf.image.resize(final_img[0], [299, 299])

    content_img_batch = tf.expand_dims(content_img_resized, axis=0)
    final_img_batch = tf.expand_dims(final_img_resized, axis=0)

    content_img_batch = tf.image.convert_image_dtype(content_img_batch, dtype=tf.float32)
    final_img_batch = tf.image.convert_image_dtype(final_img_batch, dtype=tf.float32)

    content_activations_dict = tfgan.eval.run_inception(content_img_batch)
    final_activations_dict = tfgan.eval.run_inception(final_img_batch)

    content_activations = content_activations_dict['pool_3'][0]
    final_activations = final_activations_dict['pool_3'][0]

    content_activations = tf.reshape(content_activations, (content_activations.shape[0], -1))
    final_activations = tf.reshape(final_activations, (final_activations.shape[0], -1))

    kid_score = tfgan.eval.kernel_classifier_distance_and_std_from_activations(
        content_activations,
        final_activations
    )

    print(f'KID: {kid_score[0]} ± {kid_score[1]}')
