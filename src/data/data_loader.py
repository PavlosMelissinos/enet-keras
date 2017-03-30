import cv2
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from utils import normalized


def load_data(dataset, data_dir, batch_size, nc, target_hw=(256, 256), data_type='train2014', shuffle=False, sample_size=None):
    img_txt = os.path.join(data_dir, data_type, 'images.txt')
    lbl_txt = os.path.join(data_dir, data_type, 'labels.txt')
    with open(img_txt) as imgdata, open(lbl_txt) as lbldata:
        imgfiles = [line.rstrip('\n') for line in imgdata]
        lblfiles = [line.rstrip('\n') for line in lbldata]
        assert len(imgfiles) == len(lblfiles)
        files = zip(imgfiles, lblfiles)

    if sample_size:
        assert isinstance(sample_size, int)
        np.random.shuffle(files)
        files = files[:sample_size]
    remainder = len(files)%batch_size
    assert remainder >= 0

    if remainder > 0:
        yield len(files) - remainder + batch_size
    else:
        yield len(files)
    rgbs = []
    lbls = []
    # batch_weights = []

    cid_to_id_mapper = dataset.cids_to_ids_map()
    mp = np.arange(0, max(dataset.ids()) + 1)
    mp[cid_to_id_mapper.keys()] = cid_to_id_mapper.values()

    while True:
        remainder = len(files)%batch_size
        assert remainder >= 0
        if shuffle:
            np.random.shuffle(files)
        if remainder > 0:  # Procrustes
            files = files + files[:batch_size-remainder]
        assert len(files) % batch_size == 0
        for imgfile, lblfile in files:
            img = cv2.imread(os.path.join(data_dir, data_type, imgfile))
            img = cv2.resize(img, target_hw[:2], interpolation=cv2.INTER_NEAREST)
            img = normalized(img)
            rgbs.append(img)

            # load label image, keep a single channel (all three should be the same)
            # TODO: make this more robust/fast/efficient
            lbl = cv2.imread(os.path.join(data_dir, data_type, lblfile))[:, :, 0]
            lbl = cv2.resize(lbl, target_hw[:2], interpolation=cv2.INTER_NEAREST)
            lbl = mp[lbl]
            assert np.max(lbl) < len(dataset.ids())
            # print('lbl shape', lbl.shape)
            # sample_weights = np.full(lbl.shape, 0.5)
            # print('sample weights shape', sample_weights.shape)
            # sample_weights[lbl > 0] = 1.5 # hack just for mscoco
            # batch_weights.append(sample_weights.ravel())

            lbl = to_categorical(lbl, nb_classes=nc)
            lbls.append(lbl)

            if len(rgbs) == batch_size:
                yield np.array(rgbs), np.array(lbls)#, np.array(batch_weights)
                rgbs = []
                lbls = []
                # batch_weights = []
