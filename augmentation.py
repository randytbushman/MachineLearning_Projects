import tensorflow as tf
import numpy as np


def mixup(x, y, alpha=0.2):
    # Generate a random index to shuffle the data
    index = tf.random.shuffle(tf.range(len(x)))

    # Generate a random value of alpha from a beta distribution
    lam = np.random.beta(alpha, alpha, len(x)).reshape(len(x), 1, 1, 1)

    # Mix the data
    mixed_x = tf.math.multiply(x, lam) + tf.math.multiply(tf.gather(x, index), 1 - lam)
    mixed_y = tf.math.multiply(y, lam) + tf.math.multiply(tf.gather(y, index), 1 - lam)

    return mixed_x, mixed_y

def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
    """
    Add salt and pepper noise to an image.

    Args:
    - image: numpy array of shape (height, width, channels) representing the image
    - salt_vs_pepper_ratio: float value representing the ratio of salt vs. pepper noise
    - amount: float value representing the amount of noise to add 0 <= amount <= 1

    Returns:
    - image with added salt and pepper noise
    """

    # Copy the original image
    noisy_image = np.copy(image)

    # Determine the number of salt and pepper pixels to add
    num_salt_pixels = int(np.ceil(amount * image.size * salt_vs_pepper_ratio))
    num_pepper_pixels = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper_ratio)))

    # Add salt noise
    coords = [np.random.randint(0, i-1, int(num_salt_pixels/image.shape[-1])) for i in image.shape[:-1]]
    noisy_image[coords[0], coords[1], :] = 1

    # Add pepper noise
    coords = [np.random.randint(0, i-1, int(num_pepper_pixels/image.shape[-1])) for i in image.shape[:-1]]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image


def cutout(image, size=16):
    """
    Randomly masks out a rectangular region of the given image.

    Args:
    - image: numpy array of shape (height, width, channels) representing the image
    - size: integer value representing the size of the mask

    Returns:
    - image with a randomly masked out rectangular region
    """
    h, w, c = image.shape
    x = np.random.randint(0, w - size)
    y = np.random.randint(0, h - size)
    image[y:y+size, x:x+size, :] = 0
    return image