# coding=utf-8
from __future__ import absolute_import, division
import os
from typing import Union

from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageOps


# I/O

def files_under(path: Path):
    for f in path.glob("*"):
        if f.is_file():
            yield f


def basename_without_ext(path_to_file: Path):
    return path_to_file.stem


def ensure_dir(dir_path: Union[Path, str]):
    """
    Creates folder f if it doesn't exist
    :param dir_path: directory path
    :return: 
    """
    path = dir_path if isinstance(dir_path, Path) else Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def normalize(img):
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


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def unzip_and_remove(zipped_file):
    import zipfile

    outpath = os.path.dirname(os.path.realpath(zipped_file))
    with open(zipped_file, 'rb') as fin:
        z = zipfile.ZipFile(file=fin)
        z.extractall(outpath)
        z.close()
    # os.remove(zipped_file)


# Temporary place for data preprocessing pipeline

def preprocess_image(img):
    # TODO: Populate with actual logic
    # TODO: move away from here into a dedicated class (like ImageDataGenerator)
    # img = normalize(img, mode, target_type='numpy')
    # return img

    def standardize(img, minval=0, maxval=1):
        # normalize to [minval, maxval]
        standardized = img - np.min(img)
        standardized = (maxval - minval) * standardized / np.max(standardized)
        standardized += minval
        return standardized

    # img = standardize(img, minval=-1, maxval=1)
    return img


# def preprocess_label(lbl, mapper, nc, mode, keep_aspect_ratio=False):
def preprocess_label(label):
    """
    load label image, keep a single channel (all three should be the same)
    :param label:
    :return:
    """
    # TODO: Populate with actual logic and move away from here into a dedicated class (like ImageDataGenerator)
    # target = 'pillow' if mode == 'pillow' else 'numpy'
    # # lbl = resize(lbl, target_h, target_w, mode=mode, target_type='numpy', keep_aspect_ratio=keep_aspect_ratio)
    # if mode == 'pillow':
    #     # lbl = np.expand_dims(lbl[:, :, 0], axis=2)
    #     assert np.all(lbl[:, :, 0] == lbl[:, :, 1]) and np.all(lbl[:, :, 0] == lbl[:, :, 2])
    #     lbl = lbl[:, :, 0].astype(np.uint8)
    # array2d = mapper[lbl]
    # onehot_lbl = to_categorical(array2d, num_classes=nc)
    # return onehot_lbl
    return label
