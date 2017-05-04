# coding=utf-8
from __future__ import absolute_import

import cv2
from skimage import io, transform
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
    d = os.path.dirname(f)
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def normalized(img, mode='cv2', target_type='numpy'):
    if mode == 'cv2':
        channels = [cv2.equalizeHist(channel_feature_map) for channel_feature_map in cv2.split(img)]
        norm = cv2.merge(channels)
        return norm
    elif mode == 'pillow':
        processed_img = ImageOps.equalize(img)
        if target_type == 'numpy':
            processed_img = img_to_array(processed_img)
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


# def merge(rgb, alpha):
#     r, g, b = cv2.split(rgb)
#     if len(alpha.shape[2]) == 4:
#         _, _, _, a = cv2.split(alpha)
#     else:
#         raise NotImplementedError('Can only merge two rgba images.')
#     # elif len(alpha.shape) == 2 or len(alpha.shape[2]) == 1:
#     #     a = alpha
#     res = cv2.merge([r, g, b, a])
#     return res


# def split_alpha(b_g_r_a):
#     b, g, r, a = cv2.split(b_g_r_a)
#     bgr = cv2.merge([b, g, r])
#     return bgr, a


# def multiclass2binary(label):
#     img_new = np.zeros((label.shape[0], label.shape[1]))
#     img_new[label[:, :, 0] > 0] = 1
#     return img_new


# def rgba_to_binary(lbl, value):
#     if isinstance(value, list):
#         lbl2 = np.zeros(lbl.shape[0], lbl.shape[0], len(value))
#     else:
#         lbl2 = np.zeros(lbl.shape[0], lbl.shape[0])
#     lbl2[lbl[:, :, 3] == 255] = value
#     return lbl2


def one_hot_to_rgb(onehot, id_to_palette):
    """
    Converts a one hot label to an rgb image. Pixels that belong to more than one categories 
    are assigned the maximum class id 

    :param onehot: label in onehot representation
    :param id_to_palette: dictionary that maps a class id to rgb
    :return:
    """
    rgb_label = np.zeros((onehot.shape[0], onehot.shape[1], 3), dtype=np.uint8)
    # for dim in range(onehot.shape[2]):
    #     rgb_label[onehot[:, :, dim] == 1] += 1

    for dim in range(onehot.shape[2]):
        # rgb_label[onehot[:, :, dim] > 0] = dim
        rgb_label[onehot[:, :, dim] > 0] = id_to_palette[dim]
    return rgb_label


# misc

def load_image(img_path, lib_format='cv2', target_type='numpy'):
    """
    Loads an image from a file

    :param img_path: path to image on disk
    :param lib_format: One of 'cv2', 'pillow' or 'skimage'. Currently only cv2 and pillow formats are supported
    :param target_type: 'numpy' or 'pillow'. If format is 'pillow' and target is 'numpy' then the loaded image is 
    converted to a numpy array and vice versa.

    :return: An instance of PIL.Image target is 'pillow', numpy.ndarray otherwise.
    """
    if lib_format == 'cv2':
        img = cv2.imread(img_path)
    elif lib_format == 'pillow':
        img = PILImage.open(img_path)
    elif lib_format == 'skimage':
        img = io.imread(img_path)  # TODO: resolve hw vs wh issue
    else:
        raise NotImplementedError('Unknown format {}'.format(lib_format))

    # convert output to desired data type
    if target_type == 'numpy':
        converted_img = img_to_array(img) if isinstance(img, PILImage.Image) else img
    elif target_type == 'pillow':
        converted_img = array_to_img(img) if isinstance(img, np.ndarray) else img
    else:
        raise NotImplementedError('Unknown target type {}'.format(target_type))

    return converted_img


def resize(item, target_h, target_w, lib_format='cv2', target_type='numpy', keep_aspect_ratio=False):
    """
    Resizes an image to match target dimensions
    :param item: 3d numpy array or PIL.Image
    :param target_h: height in pixels (integer)
    :param target_w: width in pixels (integer)
    :param lib_format: one of 'cv2', 'pillow', or 'skimage' (temporary, kept until one is chosen for good)
    :param target_type: 'numpy' or 'pillow'
    :param keep_aspect_ratio: If False then image is rescaled to smallest dimension and then cropped
    :return: 3d numpy array unless format is 'pillow' and input is an instance of PIL.Image
    """
    if lib_format == 'cv2':
        img = img_to_array(item) if isinstance(item, PILImage.Image) else item
        if item.shape[:2] == (target_h, target_w):
            img_resized = img
        else:
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    elif lib_format == 'pillow':
        img = array_to_img(item) if isinstance(item, np.ndarray) else item
        if keep_aspect_ratio:
            img_resized = img.thumbnail((target_w, target_w), PILImage.ANTIALIAS)
        else:
            img_resized = img.resize((target_w, target_h), resample=PILImage.BICUBIC)
    elif lib_format == 'skimage':
        img = img_to_array(item) if isinstance(item, PILImage.Image) else item
        img_resized = transform.resize(img, output_shape=(target_h, target_w))
    else:
        raise NotImplementedError()

    # convert output
    if target_type == 'numpy' and isinstance(img_resized, PILImage.Image):
        img_resized = img_to_array(img_resized)
        img_resized.astype(np.uint8)
    elif target_type == 'pillow' and isinstance(img_resized, np.ndarray):
        img_resized = array_to_img(img_resized)

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


# # TODO: check validity
# def split_pngs(finame, foname):
#     lines = []
#     directory = os.path.dirname(finame)
#     with open(finame, 'r') as fin:
#         for line in fin:
#             line = line.rstrip('\n')
#             line = os.path.join(directory, line)
#             bgra = cv2.imread(line, cv2.IMREAD_UNCHANGED)
#             bgr = bgra[:, :, :3]
#             a = bgra[:, :, 3]
#             a[a > 0] = 1
#             line = line.replace('interim', 'processed')
#             line = [line, line.replace('.png', '_lbl.png')]
#             ensure_dir(os.path.dirname(line[0]))
#             ensure_dir(os.path.dirname(line[1]))
#             lines.append(' '.join(line))
#
#             cv2.imwrite(line[0], bgr)
#             cv2.imwrite(line[1], a)
#
#     with open(foname, 'w') as fout:
#         fout.write('\n'.join(lines))


def clahe(img, mode='rgb'):
    """
    http://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    :param img:
    :param mode:
    :return:
    """
    if mode == 'rgb' and len(img.shape) == 3 and img.shape[2] == 3:
        img == img[:, :, ::-1]

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahed = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahed.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def clahe2(img):
    """
    not working with conda
    :param img:
    :return:
    """
    clahed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahed.apply(img)
    return cl1
