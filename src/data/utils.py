from __future__ import absolute_import

import cv2
import numpy as np
import os
from pycocotools import mask


# I/O

files_under = lambda path: [os.path.join(path, f) for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]


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


def normalized(img):
    channels = [cv2.equalizeHist(cfeatmap) for cfeatmap in cv2.split(img)]
    norm = cv2.merge(channels)
    return norm


# masks

def mask_rgb_to_gray(rgb, palette):
    rows = rgb.shape[0]
    cols = rgb.shape[1]
    gray = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            gray[r, c] = palette[rgb[r, c, 0], rgb[r, c, 1], rgb[r, c, 2]]
    return gray


def merge(rgb, alpha):
    r, g, b = cv2.split(rgb)
    if len(alpha.shape[2]) == 4:
        _, _, _, a = cv2.split(alpha)
    else:
        raise NotImplementedError('Can only merge two rgba images.')
    # elif len(alpha.shape) == 2 or len(alpha.shape[2]) == 1:
    #     a = alpha
    res = cv2.merge([r, g, b, a])
    return res


def multiclass2binary(label):
    img_new = np.zeros((label.shape[0], label.shape[1]))
    img_new[label[:, :, 0] > 0] = 1
    return img_new


def rgba_to_binary(lbl, value):
    if isinstance(value, list):
        lbl2 = np.zeros(lbl.shape[0], lbl.shape[0], len(value))
    else:
        lbl2 = np.zeros(lbl.shape[0], lbl.shape[0])
    lbl2[lbl[:, :, 3] == 255] = value
    return lbl2


def split_alpha(bgra):
    b, g, r, a = cv2.split(bgra)
    bgr = cv2.merge([b, g, r])
    return bgr, a


# TODO: check validity
def split_pngs(finame, foname):
    lines = []
    directory = os.path.dirname(finame)
    with open(finame, 'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            line = os.path.join(directory, line)
            bgra = cv2.imread(line, cv2.IMREAD_UNCHANGED)
            bgr = bgra[:, :, :3]
            a = bgra[:, :, 3]
            a[a > 0] = 1
            line = line.replace('interim', 'processed')
            line = [line, line.replace('.png', '_lbl.png')]
            ensure_dir(os.path.dirname(line[0]))
            ensure_dir(os.path.dirname(line[1]))
            lines.append(' '.join(line))

            cv2.imwrite(line[0], bgr)
            cv2.imwrite(line[1], a)

    with open(foname, 'w') as fout:
        fout.write('\n'.join(lines))
