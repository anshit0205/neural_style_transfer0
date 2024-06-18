import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from utils import psnr, load_img_and_preprocess, unpreprocess, scale_img
from models import VGG16_Avgpool
from losses import gram_matrix, style_loss, total_variation_loss, compute_loss_and_grads, get_loss_and_grads_wrapper, minimize_with_lbfgs
import lpips
import torch
import tensorflow_gan as tfgan

if __name__ == '__main__':
    content_img_path = 'images/content/content_image.png'
    style_img_path = 'images/style/style_image.png'

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
    content_layer = vgg.get_layer(content_layer_name).output
    content_model = tf.keras.Model(vgg.input, content_layer)
    content_target = tf.constant(content_model.predict(content_img))

    style_layers = [vgg.get_layer(layer_name).output for layer_name in style_layer_names]
    style_model = tf.keras.Model(vgg.input, style_layers)
    style_layers_outputs = [tf.constant(output) for output in style_model.predict(style_img)]

    final_img = minimize_with_lbfgs(get_loss_and_grads_wrapper, 11, batch_shape)

    content_img_np = unpreprocess(content_img[0])
    style_img_np = unpreprocess(style_img[0])

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(scale_img(content_img_np))
    plt.title('Content Image')

    plt.subplot(1, 2, 2)
    plt.imshow(scale_img(style_img_np))
    plt.title('Style Image')

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(scale_img(final_img[0]))
    plt.title('Result Image')
    plt.show()

    content_gray = tf.image.rgb_to_grayscale(content_img_np)
    result_gray = tf.image.rgb_to_grayscale(final_img[0])
    ssim_index = ssim(content_gray.numpy().squeeze(), result_gray.numpy().squeeze(), data_range=result_gray.numpy().max() - result_gray.numpy().min())
    print(f'SSIM: {ssim_index}')

    psnr_value = psnr(content_img_np, final_img[0])
    print(f'PSNR: {psnr_value} dB')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(content_img_np.ravel(), bins=256, color='red', alpha=0.5)
    plt.title('Content Image Histogram')

    plt.subplot(1, 3, 2)
    plt.hist(style_img_np.ravel(), bins=256, color='blue', alpha=0.5)
    plt.title('Style Image Histogram')

    plt.subplot(1, 3, 3)
    plt.hist(final_img.ravel(), bins=256, color='green', alpha=0.5)
    plt.title('Result Image Histogram')

    plt.show()

    lpips_fn = lpips.LPIPS(net='alex')

    content_img_tensor = torch.tensor(content_img_np).permute(2, 0, 1).unsqueeze(0)
    final_img_tensor = torch.tensor(final_img[0].copy().transpose(2, 0, 1)).unsqueeze(0)

    content_img_tensor = content_img_tensor.float()
    final_img_tensor = final_img_tensor.float()

    lpips_score = lpips_fn(content_img_tensor, final_img_tensor)
    print(f'LPIPS: {lpips_score.item()}')

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
