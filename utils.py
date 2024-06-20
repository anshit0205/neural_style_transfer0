import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

def display_images(content_img_np, style_img_np, final_img):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(scale_img(content_img_np))
    plt.title('Content Image')

    plt.subplot(1, 2, 2)
    plt.imshow(scale_img(style_img_np))
    plt.title('Style Image')

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(scale_img(final_img))
    plt.title('Result Image')
    plt.show()

def display_histograms(content_img_np, style_img_np, final_img):
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
