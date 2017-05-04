# coding=utf-8
from __future__ import print_function, division, absolute_import
from . import datasets
from .utils import one_hot_to_rgb
from keras.preprocessing.image import array_to_img
import os
import sys


def extract_coco_labels(data_dir, data_type, target_dir):
    if data_type not in ['train2014', 'val2014']:
        raise ValueError('Data type {} not recognized'.format(data_type))
    target_dir = data_dir if not target_dir else target_dir
    mscoco = datasets.load('mscoco')
    coco = mscoco.load(data_dir, data_type)

    total = len(coco.imgs)
    idx = 0
    for res in mscoco.sample_generator(coco):
        if not res:
            status = 'Skip'
        else:
            # convert label to rgb
            img, mask = res[0], res[1]
            id_to_palette = mscoco.id_to_palette_map()
            rgb_label = one_hot_to_rgb(mask, id_to_palette)

            # extract target filename
            filename_no_ext = os.path.splitext(img['file_name'])[0]
            lbl_path = os.path.join(target_dir, data_type, 'labels', '{}.png'.format(filename_no_ext))

            # convert array to PIL Image and save image to disk in png format (lossless)
            label = array_to_img(rgb_label)
            label.save(lbl_path)
            status = 'OK'
        idx += 1
        print('Processed {} out of {} images. Status: {}'.format(idx, total, status), end='\r')
        sys.stdout.flush()


if __name__ == "__main__":
    data_dir = sys.argv[1]
    data_type = sys.argv[2]
    target_dir = sys.argv[3] if len(sys.argv) > 3 else None

    extract_coco_labels(data_dir, data_type, target_dir)
