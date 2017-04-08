from __future__ import absolute_import
from __future__ import print_function

import cv2
from PIL import Image as PILImage
from skimage.transform import resize as sk_resize
from skimage import io
import numpy as np
import os
import sys

from data import mscoco as dataset
from data.utils import normalized, basename_without_ext
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from models import autoencoder


def color_output_image(dataset, img):
    colormap = dataset.id_to_palette_map()
    cvimg = np.zeros((img.shape[0], img.shape[1], 3))
    # cvimg = np.zeros((img.size[0], img.size[1], 3), dtype=np.uint8)
    # img = np.asarray(img, dtype=np.uint8).reshape(img.size[0], img.size[1])

    cvimg[img > 0] = 255
    # for cid, color in colormap.iteritems():
    #     cvimg[img == cid] = color
    return cvimg


def pillow_invert_channels(img):
    r, g, b = img.split()
    img = PILImage.merge("RGB", (b, g, r))
    return img


def predict_pillow(segmenter, imfile, h, w):
    np.random.seed(1337)  # for reproducibility
    try:
        img = load_img(imfile)
        ow, oh = img.size
        img = img.resize((w, h))

        # temporary until cv2 is replaced and the network is retrained with rgb inputs
        img = pillow_invert_channels(img)

        if img is None:
            return None

        img = img_to_array(img).astype(np.uint8)
        img = np.expand_dims(normalized(img), axis=0)
        pred = segmenter.predict(img)
        pred = np.argmax(pred[0], axis=1)
        pred = np.reshape(pred, (h, w))
        # pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
        pred = PILImage.fromarray(pred, 'RGB')
        pred = array_to_img(pred)
        pred = pred.resize((ow, oh))
        pred = img_to_array(pred)
        pred = color_output_image(dataset, pred)
        # pred = PILImage.fromarray(pred, 'RGB')
        # pred = pred.resize((ow, oh))
        return pred
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise


def predict_opencv(segmenter, imfile, h, w):
    np.random.seed(1337)  # for reproducibility
    try:
        img = cv2.imread(imfile)
        ow, oh = img.shape[:2]

        if img is None:
            return None

        img = cv2.resize(np.array(img, dtype=np.uint8), (w, h))
        img = np.expand_dims(normalized(img), axis=0)
        pred = segmenter.predict(img)
        pred = np.argmax(pred[0], axis=1)
        pred = np.reshape(pred, (h, w))
        pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
        pred = color_output_image(dataset, pred)
        return pred
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise


def predict_skimage(segmenter, imfile, h, w):
    np.random.seed(1337)  # for reproducibility
    try:
        img = io.imread(imfile)
        oh, ow = img.shape
        img = sk_resize((w, h))

        # temporary until cv2 is replaced and the network is retrained with rgb inputs
        img = img[:, :, ::-1]

        if img is None:
            return None

        img = np.expand_dims(normalized(img), axis=0)
        pred = segmenter.predict(img)
        pred = np.argmax(pred[0], axis=1)
        pred = np.reshape(pred, (h, w))
        pred = sk_resize(pred, (ow, oh))
        pred = color_output_image(dataset, pred)
        return pred
    except:
        if img is None:
            print('Skipping corrupted image')
            return None
        else:
            raise

if __name__ == '__main__':
    mode = 'opencv'
    filetxt = sys.argv[1]  # txt with image filenames
    pw = sys.argv[2]  # pretrained weights
    out_directory = sys.argv[3]  # output directory
    # dw = 256
    # dh = 256
    # dw = None
    # dh = None
    dw = 480
    dh = 480
    nc = len(dataset.ids())

    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        # ss.run(K.tf.global_variables_initializer())

    # pw = os.path.join(os.path.dirname(solver_json), solver['pw'])
    with open(filetxt, 'r') as fin:
        basedir = os.path.dirname(filetxt)
        files = [os.path.join(basedir, line.rstrip('\n')) for line in fin]

    print(pw)

    segmenter, model_name = autoencoder(nc=nc, input_shape=(dh, dw))
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

        if mode == 'pillow':
            pred_final = predict_pillow(segmenter, imfile, h=dh, w=dw)
            # pred_final.save(out_file)
        elif mode == 'opencv':
            pred_final = predict_opencv(segmenter, imfile, h=dh, w=dw)
            # res = cv2.imwrite(out_file, pred_final)
        elif mode == 'skimage':
            pred_final = predict_skimage(segmenter, imfile, h=dh, w=dw)
            # res = io.imsave(out_file, pred_final)
        # break

    print('')
