# coding=utf-8
from __future__ import absolute_import, print_function

from PIL import Image as PILImage
import numpy as np
import os
import six
import sys

from keras import backend as K
from keras.preprocessing.image import array_to_img

from src.data import datasets, utils
from src.experiments.core import Experiment
import models


def color_output_image(colormap, img, mode='bw'):
    """
    move this into datasets.py
    :param dataset: 
    :param img: 
    :param mode: 
    :return: 
    """
    cv_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if mode == 'bw':
        cv_image[img > 0] = 255
    elif mode == 'class_palette':
        img = np.asarray(img, dtype=np.uint8).reshape(img.size[0], img.size[1])
        for cid, color in six.iteritems(colormap):
            cv_image[img == cid] = color
    else:
        raise ValueError('Unknown coloring mode: Expected one of {}; got {}'.format(['bw', 'class_palette'], mode))
    return cv_image


def predict(segmenter, img, h=None, w=None):
    np.random.seed(1337)  # for reproducibility
    try:
        oh, ow = img.shape[0], img.shape[1]

        h = oh if h is None else h
        w = ow if w is None else w

        img = utils.resize(img, target_h=h, target_w=w)
        img = np.expand_dims(utils.normalize(img), axis=0)
        pred = segmenter.predict(img)[0]
        nc = pred.shape[-1]
        scores = np.max(pred, axis=1)
        pred = np.argmax(pred, axis=1)

        scores_per_class = [np.sum(scores[pred == c]) / np.sum(scores) for c in range(nc)]

        pred = np.reshape(pred, (h, w))  # dh x dw
        pred = np.expand_dims(pred, axis=2)  # dh x dw x 1
        # pred = np.repeat(pred, repeats=3, axis=2)  # dh x dw x 3

        pred = utils.resize(pred, target_h=oh, target_w=ow)  # oh x ow x 1 (original shape)
        pred = utils.img_to_array(pred)

        return pred, scores_per_class
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise


def load_mscoco_data():
    data_type = 'val2014'
    out_directory = os.path.join('data', 'out', model_name)
    config = {'dataset_name': 'MSCOCO',
              'data_dir': 'data/MSCOCO',
              'data_type': data_type}

    dataset = getattr(datasets, config['dataset_name'])(**config)

    instance_mode = False
    keep_context = 0.2

    if len(sys.argv) > 1:
        filetxt = sys.argv[1]
        with open(filetxt) as fin:
            basedir = os.path.dirname(filetxt)
            files = [os.path.join(basedir, line.rstrip('\n')) for line in fin]
        data_gen = (utils.load_image(imfile) for imfile in files)
    else:
        data_gen = (sample[0] for sample in
                    dataset.flow())
    data = {
        'generator': data_gen,
        'root_dir': 'data',
        'num_instances': dataset.num_instances,
        'dir_target': out_directory,
        'keep_context': keep_context,
        'dataset_name': 'MSCOCO',
        'data_type': 'val2017'
    }
    return data


def load_arbitrary_data(image_filenames=None):
    out_directory = os.path.join('data', 'out', model_name)

    def data_generator():
        for image_filename in image_filenames:
            img = PILImage.open(image_filename)
            resized_img = utils.resize(img, target_w=dw, target_h=dh)
            yield resized_img

    data = {
        'generator': data_generator(),
        'root_dir': 'data',
        'num_instances': len(image_filenames),
        'dir_target': out_directory,
        'keep_context': 0,
        'dataset_name': 'MSCOCO',
        'data_type': None
    }
    return data


def run(segmenter, data):
    data_gen = data['data_gen']
    num_instances = data['num_instances']
    out_directory = os.path.realpath(data['dir_target'])
    keep_context = data['keep_context']
    # dataset = getattr(datasets, data['dataset_name'])(**data)
    dataset = getattr(datasets, data['dataset_name'])

    for idx, image in enumerate(data_gen):
        if idx > 20:
            break
        print('Processing {} out of {}'.format(idx+1, num_instances), end='\r')

        pred_final, scores = predict(segmenter, image, h=dh, w=dw)

        # draw prediction as rgb
        pred_final = color_output_image(dataset.palette, pred_final[:, :, 0])
        pred_final = array_to_img(pred_final)

        out_file = os.path.join(
            out_directory,
            '{}_{}_{}_out.png'.format(
                idx,
                keep_context,
                utils.basename_without_ext(pw)))

        sys.stdout.flush()
        if os.path.isfile(out_file):
            continue

        utils.ensure_dir(out_directory)
        print('Saving output to {}'.format(out_file))
        pilimg = PILImage.fromarray(image.astype(np.uint8), mode='RGB')
        pilimg.save(out_file.replace('_out.png', '.png'))
        pred_final.save(out_file)


def load_data(**kwargs):
    load_mscoco = kwargs['load_mscoco']
    interim_testing = kwargs['interim_testing']

    if load_mscoco:
        data = load_mscoco_data()
    else:
        txt_file = sys.argv[1]
        image_dir = os.path.dirname(txt_file)
        with open(txt_file) as fin:
            image_filenames = [os.path.join(image_dir, line.rstrip('\n')) for line in fin]
            data = load_arbitrary_data(image_filenames=image_filenames)

        if interim_testing:
            for idx, item in enumerate(data['data_gen']):
                filename, extension = os.path.splitext(image_filenames[idx])
                out_filename = filename + '_interim_w{}_h{}'.format(w, h) + extension
                PILImage.fromarray(item).save(out_filename)
    return data


def main():
    # parameters
    kwargs = {
        'dataset_name': 'MSCOCO',
        'load_mscoco': False,
        'w': 256,
        'h': 256,
        'interim_testing': False
    }

    Experiment(**kwargs)

    segmenter = load_model(**kwargs)
    data = load_data()

    run(segmenter=segmenter, data=data)


if __name__ == '__main__':
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        sess_config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
        ss = K.tf.Session(config=sess_config)
        K.set_session(ss)

    print('This script is obsolete, please use run.py in predict mode instead.')
    # main()
