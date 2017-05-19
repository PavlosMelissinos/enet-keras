# coding=utf-8
from __future__ import absolute_import, print_function

from PIL import Image as PILImage
import numpy as np
import os
import sys

from keras import backend as K
from keras.preprocessing.image import array_to_img

from .data import datasets, utils
import models


def color_output_image(dataset, img, mode='bw'):
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
        colormap = dataset.id_to_palette_map()
        img = np.asarray(img, dtype=np.uint8).reshape(img.size[0], img.size[1])
        for cid, color in colormap.iteritems():
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
        img = np.expand_dims(utils.normalized(img), axis=0)
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


if __name__ == '__main__':
    dataset_name = 'mscoco'
    data_type = 'val2014'
    model_name = 'enet_unpooling'
    pw = os.path.join('models', dataset_name, 'enet_unpooling', 'weights', '{}_best.h5'.format(model_name))
    out_directory = os.path.join('data', 'out', model_name)

    dw = 256
    dh = 256
    dataset = datasets.load(dataset_name=dataset_name, data_dir='data/mscoco_reduced', data_type=data_type)
    nc = dataset.num_classes()

    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)

    print(pw)

    autoencoder = models.select_model(model_name=model_name)
    segmenter, model_name = autoencoder.build(nc=nc, w=dw, h=dh)
    segmenter.load_weights(pw)

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
                    dataset.sample_generator(instance_mode=instance_mode, keep_context=keep_context))
    for idx, image in enumerate(data_gen):
        if idx > 20:
            break
        print('Processing {} out of {}'.format(idx+1, dataset.num_instances), end='\r')

        pred_final, scores = predict(segmenter, image, h=dh, w=dw)

        # draw prediction as rgb
        pred_final = color_output_image(dataset, pred_final[:, :, 0])
        pred_final = array_to_img(pred_final)

        out_file = os.path.join(os.path.realpath(out_directory),
                                '{}_{}_{}_out.png'.format(idx, keep_context, utils.basename_without_ext(pw)))

        sys.stdout.flush()
        if os.path.isfile(out_file):
            continue

        utils.ensure_dir(out_directory)
        print('Saving output to {}'.format(out_file))
        pilimg = PILImage.fromarray(image.astype(np.uint8), mode='RGB')
        pilimg.save(out_file.replace('_out.png', '.png'))
        pred_final.save(out_file)
