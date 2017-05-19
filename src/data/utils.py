# coding=utf-8
from __future__ import absolute_import

from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image as PILImage, ImageOps
import errno
import numpy as np
import os


# I/O

def files_under(path):
    for f in os.listdir(path):
        item = os.path.join(path, f)
        if os.path.isfile(item):
            yield item
    # return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def basename_without_ext(path_to_file):
    bn = os.path.basename(path_to_file)
    return os.path.splitext(bn)[0]


def ensure_dir(f):
    """
    Creates folder f if it doesn't exist
    :param f: path to file
    :return: 
    """
    d = f
    # d = os.path.dirname(f)
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def normalized(img):
    if isinstance(img, np.ndarray):
        processed_img = ImageOps.equalize(PILImage.fromarray(img, mode='RGB'))
    else:
        processed_img = ImageOps.equalize(img)
    return processed_img


# masks

def mask_rgb_to_gray(rgb, palette):
    rows = rgb.shape[0]
    cols = rgb.shape[1]
    gray = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            gray[r, c] = palette[rgb[r, c, 0], rgb[r, c, 1], rgb[r, c, 2]]
    return gray


def one_hot_to_rgb(onehot, id_to_palette):
    """
    Converts a one hot label to an rgb image. Pixels that belong to more than one categories 
    are assigned the maximum class id 

    :param onehot: label in onehot representation
    :param id_to_palette: dictionary that maps a class id to rgb
    :return:
    """
    label = np.argmax(onehot, axis=2)
    rgb_label = np.repeat(np.expand_dims(label, axis=2), 3, axis=2)
    return rgb_label


def soften_targets(array, low=0.1, high=0.9):
    assert list(set(np.unique(array)) ^ {0, 1}) == [], 'Targets must be binary'
    array_new = np.empty_like(array)
    array_new = np.copyto(array_new, array)
    array_new[array == 0] = low
    array_new[array == 1] = high
    return array_new


# misc

def load_image(img_path):
    """
    Loads an image from a file

    :param img_path: path to image on disk

    :return: An instance of PIL.Image target is 'pillow', numpy.ndarray otherwise.
    """
    img = PILImage.open(img_path).convert('RGB')
    converted_img = img_to_array(img)
    return converted_img


def resize(item, target_h, target_w, keep_aspect_ratio=False):
    """
    Resizes an image to match target dimensions
    :type item: np.ndarray
    :type target_h: int
    :type target_w: int
    :param item: 3d numpy array or PIL.Image
    :param target_h: height in pixels
    :param target_w: width in pixels
    :param keep_aspect_ratio: If False then image is rescaled to smallest dimension and then cropped
    :return: 3d numpy array
    """
    img = array_to_img(item, scale=False)
    if keep_aspect_ratio:
        img.thumbnail((target_w, target_w), PILImage.ANTIALIAS)
        img_resized = img
    else:
        img_resized = img.resize((target_w, target_h), resample=PILImage.NEAREST)

    # convert output
    img_resized = img_to_array(img_resized)
    img_resized = img_resized.astype(dtype=np.uint8)

    return img_resized


def center_crop(x, y=None, crop_size=None, data_format='channels_last'):
    """
    Takes a pair of numpy arrays (image and label) and returns a pair of matching center crops
    :param x: image in numpy array format
    :param y: label in numpy array format
    :param crop_size: (height, width) tuple
    :param data_format: 'channels_first' or 'channels_last'
    :return: (cropped image, cropped label) tuple
    """
    if crop_size is None:
        return x if y is None else x, y

    if data_format == 'channels_first':
        centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
    elif data_format == 'channels_last':
        centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    else:
        raise NotImplementedError()
    crop_size = (2 * centerh, 2 * centerw) if crop_size is None else crop_size
    lh, lw = crop_size[0] // 2, crop_size[1] // 2
    rh, rw = crop_size[0] - lh, crop_size[1] - lw

    start_h, end_h = centerh - lh, centerh + rh
    start_w, end_w = centerw - lw, centerw + rw
    if data_format == 'channels_first':
        cropped_x = x[:, start_h:end_h, start_w:end_w]
        if y is None:
            return cropped_x
        else:
            cropped_y = y[:, start_h:end_h, start_w:end_w]
            return cropped_x, cropped_y
    elif data_format == 'channels_last':
        cropped_x = x[start_h:end_h, start_w:end_w, :]
        if y is None:
            return cropped_x
        else:
            cropped_y = y[start_h:end_h, start_w:end_w, :]
            return cropped_x, cropped_y


def random_crop(x, y=None, crop_size=None, data_format='channels_last', sync_seed=None):
    """
    Takes a pair of numpy arrays (image and label) and returns a pair of matching random crops
    :param x: image in numpy array format. Shape is (h, w, c) or (c, h, w), depending on data_format.
    :param y: label in numpy array format. Shape is (h, w, c) or (c, h, w), depending on data_format.
    :param crop_size: (height, width) tuple
    :param data_format: 'channels_first' or 'channels_last'
    :param sync_seed: random seed (for easier reproduction)
    :return: (cropped image, cropped label) tuple
    """
    if crop_size is None:
        return x if y is None else x, y

    np.random.seed(sync_seed)
    if data_format == 'channels_first':
        h, w = x.shape[1], x.shape[2]
    elif data_format == 'channels_last':
        h, w = x.shape[0], x.shape[1]
    else:
        raise NotImplementedError()
    rangeh = (h - crop_size[0]) // 2
    rangew = (w - crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)

    start_h, end_h = offseth, offseth + crop_size[0]
    start_w, end_w = offsetw, offsetw + crop_size[1]
    if data_format == 'channels_first':
        cropped_x = x[:, start_h:end_h, start_w:end_w]
        if y is None:
            return cropped_x
        else:
            cropped_y = y[:, start_h:end_h, start_w:end_w]
            return cropped_x, cropped_y
    elif data_format == 'channels_last':
        cropped_x = x[start_h:end_h, start_w:end_w, :]
        if y is None:
            return cropped_x
        else:
            cropped_y = y[start_h:end_h, start_w:end_w, :]
            return cropped_x, cropped_y


def pillow_invert_channels(img):
    r, g, b = img.split()
    img = PILImage.merge("RGB", (b, g, r))
    return img
