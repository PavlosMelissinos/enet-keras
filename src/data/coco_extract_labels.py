# coding=utf-8
from __future__ import print_function, division, absolute_import
from . import datasets
from .utils import one_hot_to_rgb
from keras.preprocessing.image import array_to_img
import os
import sys


def extract_coco_labels(target_dir):
    kwargs = {
        'h': 512,
        'w': 512,
        'batch_size': 2,
        'root_dir': 'data',
        'dataset_name': 'mscoco',
        'data_type': 'train2017',
        'sample_size': 0.01,
        'instance_mode': False,
        'keep_context': 0.25,
        'merge_annotations': True,
        'cover_gaps': True,
        'resize_mode': 'stretch',
    }

    dataset = datasets.MSCOCO(**kwargs)

    for idx, res in enumerate(dataset.flow()):
        if not res:
            status = 'Skip'
        else:
            # convert label to rgb
            img, mask = res[0], res[1]
            rgb_label = one_hot_to_rgb(mask, dataset.PALETTE)

            # extract target filename
            filename_no_ext = os.path.splitext(img['file_name'])[0]
            lbl_path = os.path.join(
                target_dir,
                kwargs['data_type'],
                'labels',
                '{}.png'.format(filename_no_ext)
            )

            # convert array to PIL Image and save image to disk in png format (lossless)
            label = array_to_img(rgb_label)
            label.save(lbl_path)
            status = 'OK'
        msg = 'Processed {}/{} items. Status: {}'.format(idx + 1, dataset.num_items, status)
        print(msg, end='\r')
        sys.stdout.flush()


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None

    extract_coco_labels(target_dir)
