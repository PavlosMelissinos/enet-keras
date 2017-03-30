from __future__ import absolute_import
from __future__ import print_function

import cv2
import json
import numpy as np
import os
import sys

from data import mscoco as dataset
from data.utils import normalized, basename_without_ext
from keras import backend as K
import models


def colorImageOut(dataset, img):
    # colors = [[128, 0, 0], [0,128,0], [128,128,0], [0,0,128], [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0], [0,192,0], [128,192,0], [0,64,128], [128,64,128], [0,192,128], [128,192,128], [64,64,0], [192,64,0], [64,192,0], [192,192,0], [64,64,128], [192,64,128], [64,192,128], [192,192,128], [0,0,64], [128,0,64], [0,128,64], [128,128,64], [0,0,192], [128,0,192], [0,128,192], [128,128,192], [64,0,64], [192,0,64], [64,128,64], [192,128,64], [64,0,192], [192,0,192], [64,128,192], [192,128,192], [0,64,64], [128,64,64]]
    colormap = dataset.cid_to_palette_map()
    cvimg = np.zeros((img.shape[0], img.shape[1], 3))
    for cid, color in colormap.iteritems():
        cvimg[img == cid] = color
    return cvimg


def predict(segmenter, imfile, target_shape):
    np.random.seed(1337) # for reproducibility
    try:
        img = cv2.imread(imfile)
        if img is None:
            return None
        shape_orig = img.shape
        img = cv2.resize(img, target_shape)
        img = np.asarray([img])
        pred = segmenter.predict(img)
        pred = np.argmax(pred[0], axis=1)
        pred = np.reshape(pred, (target_shape[1], target_shape[0]))
        img = cv2.resize(img, shape_orig)
        pred = colorImageOut(dataset, pred)
        return pred
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise

if __name__ == '__main__':
    filetxt = sys.argv[1]
    dw = 256
    dh = 256
    nc = len(dataset.ids())

    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    pw = os.path.join(os.path.dirname(solver_json), solver['pw'])
    with open(filetxt, 'r') as fin:
        basedir = os.path.dirname(filetxt)
        files = [os.path.join(basedir, line.rstrip('\n')) for line in fin]

    print(pw)

    segmenter, model_name = autoencoder(nc=nc, input_shape=(dw, dh))
    segmenter.load_weights(pw)
    for idx, imfile in enumerate(files):
        out_file = os.path.join(os.path.dirname(os.path.realpath(imfile)),
            '{}_{}_out.png'.format(
                basename_without_ext(imfile),
                basename_without_ext(pw)))
        print('Processing {} out of {}'.format(idx+1, len(files)), end='\r')
        sys.stdout.flush()
        if os.path.isfile(out_file):
            continue
        pred = predict(segmenter, imfile, target_shape=(dw, dh))
        res = cv2.imwrite(out_file, pred)
    print('')
