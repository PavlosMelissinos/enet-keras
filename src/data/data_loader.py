# coding=utf-8
from __future__ import absolute_import, division, print_function

import json
import numpy as np
import os
import sys
import time
from matplotlib import pyplot as plt

from src.data import datasets


def batched(data_generator, batch_size):
    batches = [iter(data_generator)] * batch_size
    batches = zip(*batches)
    for batch in batches:
        yield [np.array(item) for item in zip(*batch)]


def batched_slow(data_generator, batch_size):
    images = []
    labels = []
    for image, label in data_generator:
        images.append(image)
        labels.append(label)
        if len(images) == batch_size:
            yield np.array(images), np.array(labels)
            images = []
            labels = []


# testing code for data loader
# TODO: convert into proper unit test for the function

def time_data_generator(data_generator, sample_size):
    start = time.clock()
    for idx, items in enumerate(data_generator):
        sys.stdout.flush()
        if idx >= sample_size:
            break
        print('Processed {} items: ({})'.format(idx + 1, [type(item) for item in items]), end='\r')
    print(time.clock() - start)


def test_dataset(solver):
    data_config = solver['data']
    data_config['h'] = solver['dh']
    data_config['w'] = solver['dw']
    dataset_name = data_config['dataset_name']

    print('Preparing to train on {} data...'.format(dataset_name))

    supplementary_data_config = data_config['val']
    data_config.update(supplementary_data_config)

    np.random.seed(1337)  # for reproducibility

    dataset = datasets.load(dataset_name=dataset_name)(config=data_config)

    sample_size = 10

    time_data_generator(dataset.flow(), sample_size=sample_size)


def test(solver):
    np.random.seed(1337)  # for reproducibility

    # full_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.append(full_path)

    data_config = solver['data']
    data_config['w'] = solver['dw']
    data_config['h'] = solver['dh']
    supplementery_data_config = data_config['val']
    data_config.update(supplementery_data_config)

    dataset_name = 'mscoco'

    print('Loading {} data...'.format(dataset_name))

    dataset = datasets.load(dataset_name=dataset_name)(config=data_config)
    print('Done')

    for idx, item in enumerate(dataset.flow()):
        img, lbl = item[0].astype(np.uint8), item[1]
        batch_size = img.shape[0]
        h = img.shape[1]
        w = img.shape[2]
        nc = lbl.shape[-1]
        lbl = np.reshape(lbl, (batch_size, h, w, nc))
        for batch_index in range(data_config['batch_size']):
            binary_masks = split_label_channels(lbl[batch_index, ...])
            img_item = img[batch_index, ...]
            for class_idx, binary_mask in binary_masks.items():
                # class_name = dataset.CATEGORIES[dataset.IDS[class_idx]]
                class_name = dataset.CATEGORIES[class_idx]

                plt.rcParams["figure.figsize"] = [4 * 3, 4]

                fig = plt.figure()

                subplot1 = fig.add_subplot(131)
                subplot1.imshow(img_item)
                subplot1.set_title('rgb image')
                subplot1.axis('off')

                subplot2 = fig.add_subplot(132)
                subplot2.imshow(binary_mask, cmap='gray')
                subplot2.set_title('{} binary mask'.format(class_name))
                subplot2.axis('off')

                subplot3 = fig.add_subplot(133)
                masked = np.array(img_item)
                masked[binary_mask == 0] = 0
                subplot3.imshow(masked)
                subplot3.set_title('{} label'.format(class_name))
                subplot3.axis('off')

                fig.tight_layout()
                plt.show()
        # shapes.append(img.shape)
        print('Processed {} items: ({})'.format(idx + 1, type(item)), end='\r')
        sys.stdout.flush()


def split_label_channels(label):
    binary_masks = {}
    for i in range(label.shape[-1]):
        binary_mask = label[..., i]
        if not np.any(binary_mask > 0):
            continue
        binary_mask[binary_mask > 0] = 1
        binary_masks[i] = binary_mask.astype(np.uint8)
    return binary_masks


if __name__ == '__main__':
    solver_json = 'config/solver.json'
    print('solver json: {}'.format(os.path.abspath(solver_json)))

    test(solver=json.load(open(solver_json)))
    # test_dataset(solver=json.load(open(solver_json)))
