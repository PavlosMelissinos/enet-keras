# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import sys

from data import datasets
from data.utils import normalized, basename_without_ext, pillow_invert_channels
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from models import build


def color_output_image(dataset, img, mode='bw'):
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


def predict(segmenter, image_file, h, w):
    np.random.seed(1337)  # for reproducibility
    img = None
    try:
        img = load_img(image_file)
        ow, oh = img.size
        img = img.resize((w, h))

        # TODO: temporary until opencv is replaced and the network is retrained with rgb inputs
        img = pillow_invert_channels(img)

        if img is None:
            return None

        img = img_to_array(img).astype(np.uint8)
        img = np.expand_dims(normalized(img), axis=0)
        pred = segmenter.predict(img)
        pred = np.argmax(pred[0], axis=1)
        pred = np.reshape(pred, (h, w))
        pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        pred = array_to_img(pred)
        pred = pred.resize((ow, oh))
        pred = img_to_array(pred)
        pred = color_output_image(dataset, pred[:, :, 0])
        pred = array_to_img(pred)
        return pred
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise


if __name__ == '__main__':
    filetxt = sys.argv[1]  # txt with image filenames
    pw = sys.argv[2]  # pretrained weights
    out_directory = sys.argv[3]  # output directory
    dataset_name = 'mscoco'

    dw = 480
    dh = 480
    dataset = datasets.load(dataset_name=dataset_name)
    nc = dataset.num_classes()

    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        # ss.run(K.tf.global_variables_initializer())

    # pw = os.path.join(os.path.dirname(solver_json), solver['pw'])
    with open(filetxt) as fin:
        basedir = os.path.dirname(filetxt)
        files = [os.path.join(basedir, line.rstrip('\n')) for line in fin]

    print(pw)

    segmenter, model_name = build(nc=nc, w=dw, h=dh)
    segmenter.load_weights(pw)
    for idx, imfile in enumerate(files):
        # out_file = os.path.join(os.path.dirname(os.path.realpath(imfile)),
        out_file = os.path.join(os.path.realpath(out_directory),
                                '{}_{}_out.png'.format(
                                    basename_without_ext(imfile),
                                    basename_without_ext(pw)))
        print('Processing {} out of {}'.format(idx+1, len(files)), end='\r')
        sys.stdout.flush()
        if os.path.isfile(out_file):
            continue

        pred_final = predict(segmenter, imfile, h=dh, w=dw)
        pred_final.save(out_file)
