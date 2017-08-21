# coding=utf-8
from __future__ import print_function, absolute_import, division

import numpy as np
import os
from . import datasets, utils


def collect_image_files_from_disk(data_dir, data_type, sample_size=None):
    """
    load file list from disk in pairs
    :param data_dir: 
    :param data_type: 
    :param sample_size: 
    :return: 
    """
    img_txt = os.path.join(data_dir, data_type, 'images.txt')
    lbl_txt = os.path.join(data_dir, data_type, 'labels.txt')
    with open(img_txt) as image_data, open(lbl_txt) as label_data:
        image_files = [os.path.join(data_dir, data_type, line.rstrip('\n')) for line in image_data]
        label_files = [os.path.join(data_dir, data_type, line.rstrip('\n')) for line in label_data]
        assert len(image_files) == len(label_files)
        files = zip(image_files, label_files)

    if sample_size:
        assert isinstance(sample_size, int)
        np.random.shuffle(files)
        files = files[:sample_size]
    return files


# def soften_targets(array, low=0.1, high=0.9):
#     assert list(set(np.unique(array)) ^ {0, 1}) == [], 'Targets must be binary'
#     array_new = np.empty_like(array)
#     array_new = np.copyto(array_new, array)
#     array_new[array == 0] = low
#     array_new[array == 1] = high
#     return array_new


# def preprocess_image(img, mode):
def preprocess_image(img):
    # img = normalized(img, mode, target_type='numpy')
    # return img
    return img


# def preprocess_label(lbl, mapper, nc, mode, keep_aspect_ratio=False):
def preprocess_label(label):
    """
    load label image, keep a single channel (all three should be the same)
    :param label:
    :return:
    """
    # TODO: make this more robust/fast/efficient
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


def load_dataset(source='json', dataset_name=None, data_dir=None, data_type=None,
                 sample_size=None,
                 instance_mode=True,
                 keep_context=0.25):
    if source == 'disk':
        file_pairs = collect_image_files_from_disk(data_dir, data_type, sample_size)
        dataset = None
        generator = ((utils.load_image(img_path), utils.load_image(lbl_path)) for img_path, lbl_path in file_pairs)
    elif source == 'json':
        dataset = datasets.load(dataset_name, data_dir=data_dir, data_type=data_type)
        generator = dataset.sample_generator(sample_size=sample_size, instance_mode=instance_mode, keep_context=keep_context)
    else:
        raise NotImplementedError
    return dataset, generator


def load_data(dataset=None,
              generator=None,
              target_h=None, target_w=None,
              resize_mode='stretch'):
    forced_target_h = target_h
    forced_target_w = target_w
    for img, lbl in generator:
        if not forced_target_h:
            target_h = img.shape[0]
        if not forced_target_w:
            target_w = img.shape[1]

        if target_h > lbl.shape[0] and resize_mode != 'stretch':
            continue
        if target_w > lbl.shape[1] and resize_mode != 'stretch':
            continue

        image = preprocess_image(img)
        label = preprocess_label(lbl)

        assert image.shape[:2] == label.shape[:2]

        if resize_mode == 'random_crop':
            resized_image, resized_label = utils.random_crop(x=image, y=label, crop_size=(target_h, target_w))
        elif resize_mode == 'center_crop':
            resized_image, resized_label = utils.center_crop(x=image, y=label, crop_size=(target_h, target_w))
        elif resize_mode == 'stretch':
            # TODO assumes that each label contains annotation for a single object
            resized_image = utils.resize(item=image, target_h=target_h, target_w=target_w)

            # for sample in range(label.shape[0]):
            #     label_resized = []
            #     for cid in range(label.shape[2]):
            #         label_resized.append(utils.resize(label[sample][:][cid], target_h=target_h, target_w=target_w))

            # label = utils.one_hot_to_rgb(label, dataset.palette)
            label = np.argmax(label, axis=2)
            label = np.expand_dims(label, axis=2)
            if label.dtype != np.uint8:
                label = label.astype(np.uint8)

            resized_label = utils.resize(item=label, target_h=target_h, target_w=target_w)
            resized_label = resized_label[:, :, 0].astype(dtype=np.uint8)
            resized_label = np.eye(dataset.num_classes())[resized_label]  # convert to one hot (h, w, c)

            # h_ratio = target_h / label.shape[0]
            # w_ratio = target_w / label.shape[1]
            # resized_label = scipy.ndimage.zoom(label, (h_ratio, w_ratio, 1), order=0)
        else:
            raise NotImplementedError('unknown resize mode {}'.format(resize_mode))
        assert resized_image.shape[:2] == resized_label.shape[:2]
        # resized_image = np.expand_dims(resized_image, axis=0)
        # resized_label = np.expand_dims(resized_label, axis=0)  # convert label shape to (1, h, w, c)
        assert len(resized_image.shape) == 3
        assert len(resized_label.shape) == 3
        yield resized_image, resized_label


def batched(data_generator, batch_size, flatten=True):
    images = []
    labels = []

    counter = 0
    for image, label in data_generator:
        images.append(image)
        labels.append(label)
        if len(images) == batch_size:
            counter += 1
            if flatten:
                data_shape = labels[0].shape[0] * labels[0].shape[1]
                nc = labels[0].shape[2]
                # labels = np.array(labels)
                # labels = np.rollaxis(np.dstack(labels), -1)
                labels = np.concatenate(labels, axis=0)
                labels = np.reshape(labels, (batch_size, data_shape, nc))
            images = np.array(images)
            labels = np.array(labels)
            yield np.array(images), np.array(labels)  # , np.array(batch_weights)
            images = []
            labels = []


# testing code for data loader
# TODO: convert into proper unit test for the function

def test():
    np.random.seed(1337)  # for reproducibility

    import sys
    import json

    full_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(full_path)

    solver_json = '../../config/solver.json'

    print('solver json: {}'.format(os.path.abspath(solver_json)))
    solver = json.load(open(solver_json))
    batch_size = solver['batch_size']
    dw = solver['dw']
    dh = solver['dh']

    dataset_name = 'mscoco'

    # nc = len(dataset.ids())  # categories + background

    print('Preparing to train on {} data...'.format(dataset_name))

    dataset, generator = load_dataset(dataset_name=dataset_name,
                                      data_dir=os.path.join('../../data', dataset_name),
                                      data_type='train2014')

    train_gen = load_data(dataset=dataset,
                          generator=generator,
                          # batch_size=batch_size,
                          # nc=nc,
                          # shuffle=True,
                          target_h=dh, target_w=dw)
    samples = train_gen.next()
    train_gen = batched(data_generator=train_gen, batch_size=batch_size)
    print(samples)
    for idx, item in enumerate(train_gen):
        img, lbl = item[0], item[1]
        # shapes.append(img.shape)
        print('Processed {} items: ({})'.format(idx + 1, type(item)), end='\r')
        sys.stdout.flush()

if __name__ == '__main__':
    test()
