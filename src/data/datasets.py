# coding=utf-8
from __future__ import absolute_import, division, print_function
from six import with_metaclass
import abc
import numpy as np
import os
import subprocess
import time

from pycocotools import mask
from pycocotools.coco import COCO
from . import utils


class Dataset(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def flow(self, transform=True, batch=True, single_pass=False,
             *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def steps(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def num_classes():
        pass


class CaptioningDataset(Dataset):
    @property
    @abc.abstractmethod
    def vocab(self):
        pass


class MSCOCO(Dataset):
    class Configurator(object):
        def __init__(self, dataset_name, root_dir, data_type, batch_size, h, w,
                     instance_mode,
                     sample_size=None,
                     area_threshold=2500,
                     keep_context=0.25,
                     merge_annotations=True,
                     cover_gaps=True,
                     resize_mode='stretch',
                     **kwargs
                     ):
            self.data_root = os.path.abspath(root_dir)
            self.dataset_name = dataset_name

            # setup train data location
            valid_data_types = [
                'train2014',
                'val2014',
                'test2014',
                'test2015',
                'train2017',
                'val2017',
                'test2017'
            ]
            if data_type not in valid_data_types:
                errmsg = 'Unknown data type {}. Valid values are {}'.format(
                    data_type,
                    valid_data_types
                )
                raise ValueError(errmsg)
            self.data_type = data_type

            try:
                dataset_root = os.path.join(self.data_root, self.dataset_name)
            except:
                self.data_root = os.path.join(os.path.expanduser('~'),
                                              '.datasets')
                dataset_root = os.path.join(self.data_root,
                                            self.dataset_name)
            self.annotation_file = os.path.join(dataset_root,
                                                'annotations',
                                                'instances_' + self.data_type + '.json')
            self.image_dir = os.path.join(dataset_root, self.data_type)

            self.data_dir = {
                'dataset_root': dataset_root,
                'annotations': self.annotation_file,
                'images': self.image_dir,
            }

            self.instance_mode = instance_mode

            sample_size = 1 if not sample_size else sample_size
            if sample_size <= 1:
                self.sample_factor = sample_size
            else:
                self.sample_size = sample_size
            self.area_threshold = area_threshold

            # flow parameters, TODO: enable modification later
            self.keep_context = keep_context
            self.merge_annotations = merge_annotations
            self.cover_gaps = cover_gaps
            self.resize_mode = resize_mode
            self.batch_size = batch_size
            self.target_height = h
            self.target_width = w

    CATEGORIES = [
        'background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    IDS = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    PALETTE = [(cid, cid, cid) for cid in range(max(IDS) + 1)]

    # CLASS_FREQUENCIES = [
    #     1.0000,
    #     0.0182, 0.0132, 0.0134, 0.0134, 0.0144, 0.0143, 0.0143, 0.0133,
    #     0.0144, 0.0093, 0.0181, 0.0093, 0.0114, 0.0089, 0.0229, 0.0241,
    #     0.0243, 0.0258, 0.0255, 0.0256, 0.0254, 0.0241, 0.0258, 0.0254,
    #     0.0130, 0.0154, 0.0128, 0.0176, 0.0137, 0.0105, 0.0092, 0.0181,
    #     0.0071, 0.0040, 0.0068, 0.0068, 0.0181, 0.0181, 0.0083, 0.0138,
    #     0.0132, 0.0128, 0.0167, 0.0166, 0.0166, 0.0129, 0.0002, 0.0001,
    #     0.0027, 0.0003, 0.0005, 0.0005, 0.0028, 0.0025, 0.0008, 0.0028,
    #     0.0089, 0.0089, 0.0128, 0.0086, 0.0088, 0.0089, 0.0123, 0.0100,
    #     0.0123, 0.0091, 0.0121, 0.0073, 0.0179, 0.0180, 0.0179, 0.0106,
    #     0.0179, 0.0039, 0.0113, 0.0141, 0.0080, 0.0040, 0.0180, 0.0083
    # ]

    # CLASS_FREQUENCIES = [0.1] + [1 / 80] * 80
    CLASS_FREQUENCIES = [1.0] * 81

    def __init__(self, **kwargs):
        """
        Accepts as input a dictionary that looks like this:

        ```
        {
          "h": 512,
          "w": 512,
          "batch_size": 2,
          "root_dir": "data",
          "dataset_name": "mscoco",
          "sample_size": 0.01,
          "instance_mode": false,
          "keep_context": 0.25,
          "merge_annotations": true,
          "cover_gaps": true,
          "resize_mode": "stretch",
          "train": {
            "data_type": "train2017"
          },
          "val": {
            "data_type": "val2017"
          },
          "test": {
            "data_type": "test2017"
          }
        }
        ```

        :param kwargs: dictionary that stores dataset parameters, check docstring example
        """
        self._config = MSCOCO.Configurator(**kwargs)

        # ensure data exists locally and load dataset
        ann_file = self._config.annotation_file
        if not os.path.isfile(ann_file):
            print('Dataset not found. Aborting...')
            exit()
            # self.download()
        print('Initializing MS-COCO: Loading annotations from {}'.format(ann_file))
        self._coco = COCO(ann_file)

        # build indices
        self._cid_to_id = {cid: idx for idx, cid in enumerate(self.IDS)}
        self._category_to_id = {category: idx
                                for idx, category in enumerate(self.CATEGORIES)}
        self._palette_to_id = {category: idx
                               for idx, category in enumerate(self.CATEGORIES)}

        # pass through the dataset and count all valid samples
        # TODO: This should be moved to a method and allowed to be executed at any point
        sample_counter = 0
        for idx, img_id in enumerate(self._coco.getImgIds()):
            annotation_ids = self._coco.getAnnIds(imgIds=img_id)
            for annotation in self._coco.loadAnns(annotation_ids):
                if annotation['area'] > self.config.area_threshold:
                    sample_counter += 1
        self._num_instances = sample_counter

        self._num_items = self.num_instances if self.config.instance_mode else self.num_images
        if hasattr(self.config, 'sample_size'):
            self._sample_size = min(self._num_items, self.config.sample_size)
        else:
            assert hasattr(self.config, 'sample_factor')
            self._sample_size = int(self.config.sample_factor * self._num_items)

        image_ids = self._coco.getImgIds()
        self._image_ids = np.random.choice(image_ids,
                                           size=min(len(image_ids), self._sample_size),
                                           replace=False)

    @property
    def categories(self):
        return MSCOCO.CATEGORIES

    @property
    def config(self):
        return self._config

    @property
    def palette(self):
        return MSCOCO.PALETTE

    @property
    def num_instances(self):
        return self._num_instances

    @property
    def num_items(self):
        return self._num_items

    @property
    def num_images(self):
        return len(self._coco.imgs)

    @property
    def steps(self):
        return self._sample_size // self._config.batch_size

    @staticmethod
    def class_frequencies():
        return MSCOCO.CLASS_FREQUENCIES

    @staticmethod
    def class_weights():
        return np.array([1/cf for cf in MSCOCO.CLASS_FREQUENCIES])

    @staticmethod
    def num_classes():
        return len(MSCOCO.IDS)

    def download(self):
        """Download MSCOCO into data_dir, verify hashes, then extract files.
        If the files are already present, only the hashes are checked.
        """
        # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
        if subprocess.call(['which', 'gsutil']) == 1:
            print('gsutil missing from your system, please install (pip install gsutil) and try again...')
            exit(1)
        data_prefixes = [self.config.data_type]

        dataset_root = self.config.data_dir['dataset_root']
        utils.ensure_dir(dataset_root)

        print('Syncing images. Please wait, this might take a while...')
        for prefix in data_prefixes:
            print(f'Downloading {prefix}...')
            url = f'gs://images.cocodataset.org/{prefix}'
            target_dir = os.path.join(dataset_root, prefix)
            utils.ensure_dir(target_dir)
            subprocess.call(['gsutil', '-m', 'rsync', url, target_dir])
        print('Done')

        print('Syncing annotations. Please wait, this might take a while...')
        ann_url = 'gs://images.cocodataset.org/annotations'

        # run shell command, the following does not work from within PyCharm
        subprocess.call(['gsutil', '-m', 'rsync', ann_url, dataset_root])

        print('Done')

        print('Extracting annotation zip archives.')
        zips = [
            'annotations_trainval2014.zip',
            'annotations_trainval2017.zip',
            'image_info_test2014.zip',
            'image_info_test2015.zip',
            'image_info_test2017.zip',
            'image_info_unlabeled2017.zip',
            'stuff_annotations_trainval2017.zip',
            'stuff_image_info_test2017.zip'
        ]
        for zip in zips:
            zipfile = os.path.join(dataset_root, zip)
            print(zipfile)
            utils.unzip_and_remove(zipped_file=zipfile)
        print('Done')

    def load(self, data_dir, data_type):
        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        print('Initializing MS-COCO: Loading annotations from {}'.format(annotation_file))
        self._coco = COCO(annotation_file)

    def _annotation_generator(self):
        """
        Generates sample_size annotations. No pre/post-processing
        :return: coco annotation (dictionary)
        """

        annotation_counter = 0

        for img_id in self._image_ids:
            annotation_ids = self._coco.getAnnIds(imgIds=img_id)
            for annotation in self._coco.loadAnns(annotation_ids):
                if annotation['area'] > self._config.area_threshold:
                    annotation_counter += 1
                    yield annotation
        if annotation_counter == 0:
            errmsg = 'Every annotation has been filtered out. ' \
                     'Decrease area threshold or increase image dimensions.'
            raise Exception(errmsg)

    def _retrieve_sample(self, annotation):
        epsilon = 0.05
        high_val = 1 - epsilon
        low_val = 0 + epsilon
        coco_image = self._coco.loadImgs(annotation['image_id'])[0]
        image_path = os.path.join(self._config.data_dir['images'], coco_image['file_name'])
        image = utils.load_image(image_path)

        ann_mask = self._coco.annToMask(annotation)

        mask_categorical = np.full((ann_mask.shape[0], ann_mask.shape[1], self.num_classes()), low_val, dtype=np.float32)
        mask_categorical[:, :, 0] = high_val  # every pixel begins as background

        class_index = self._cid_to_id[annotation['category_id']]
        mask_categorical[ann_mask > 0, class_index] = high_val
        mask_categorical[ann_mask > 0, 0] = low_val  # remove background label from pixels of this (non-bg) category
        return image, mask_categorical

    def _retrieve_instance(self, annotation):
        """
        crops a pair of image label arrays according to an annotation id
        :type annotation: dict
        """
        keep_context = self._config.keep_context
        image, mask_categorical = self._retrieve_sample(annotation=annotation)

        x, y, width, height = annotation['bbox']
        x = int(max(0., x - keep_context * width + 0.5))
        y = int(max(0., y - keep_context * height + 0.5))
        height = int(min(height + 2 * keep_context * height + 0.5, mask_categorical.shape[0]))
        width = int(min(width + 2 * keep_context * width + 0.5, mask_categorical.shape[1]))

        cropped_image = image[y: y + height, x: x + width, :]
        cropped_label = mask_categorical[y: y + height, x: x + width, :]
        return cropped_image, cropped_label

    def _combined_sample_generator(self):
        """
        Generates image/mask pairs from dataset.
        :return: generator of image-mask pairs.
        Each pair is a tuple. Image shape is (width, height, 3) and the shape of the mask is (width, height, 91)
        """

        epsilon = 0.05
        high_val = 1 - epsilon
        low_val = 0 + epsilon
        for img_id in self._image_ids:
            coco_image = self._coco.loadImgs(int(img_id))[0]

            image_path = os.path.join(self._config.data_dir['images'], coco_image['file_name'])
            image = utils.load_image(image_path)
            # TODO: maybe it's faster to resize the image here

            target_h = coco_image['height']
            target_w = coco_image['width']

            mask_one_hot = np.full((target_h, target_w, self.num_classes()), low_val, dtype=np.float32)
            mask_one_hot[:, :, 0] = high_val  # every pixel begins as background

            annotation_ids = self._coco.getAnnIds(imgIds=coco_image['id'])

            for annotation in self._coco.loadAnns(annotation_ids):
                mask_partial = self._coco.annToMask(annotation)
                assert mask_one_hot.shape[:2] == mask_partial.shape[:2]  # width and height match
                if self._config.cover_gaps:
                    class_index = self._cid_to_id[annotation['category_id']]
                else:
                    class_index = annotation['category_id']
                assert class_index > 0
                mask_one_hot[mask_partial > low_val, class_index] = high_val
                mask_one_hot[mask_partial > low_val, 0] = low_val
            yield image, mask_one_hot

    def transform(self, img, lbl, flatten_labels=True):
        resize_mode = self._config.resize_mode
        h = self._config.target_height
        w = self._config.target_width
        h = img.shape[0] if h is None else h
        w = img.shape[1] if w is None else w

        errmsg_h = 'Label height can\'t be increased in {} mode.' \
                   ' Please use stretch mode if you want zoom in' \
                   ' behaviour. Values: {} vs {}'.format(resize_mode, h, lbl.shape[0])
        errmsg_w = 'Label height can\'t be increased in {} mode.' \
                   ' Please use stretch mode if you want zoom in' \
                   ' behaviour. Values: {} vs {}'.format(resize_mode, w, lbl.shape[1])
        if h > lbl.shape[0] and resize_mode != 'stretch':
            raise ValueError(errmsg_h)
        if w > lbl.shape[1] and resize_mode != 'stretch':
            raise ValueError(errmsg_w)

        image = utils.preprocess_image(img)
        label = utils.preprocess_label(lbl)

        assert image.shape[:2] == label.shape[:2]

        if resize_mode == 'random_crop':
            resized_image, resized_label = utils.random_crop(x=image, y=label, crop_size=(h, w))
        elif resize_mode == 'center_crop':
            resized_image, resized_label = utils.center_crop(x=image, y=label, crop_size=(h, w))
        elif resize_mode == 'stretch':
            # TODO assumes that each label contains annotation for a single object
            resized_image = utils.resize(item=image, target_h=h, target_w=w)

            label = np.argmax(label, axis=2)
            label = np.expand_dims(label, axis=2)
            if label.dtype != np.uint8:
                label = label.astype(np.uint8)

            resized_label = utils.resize(item=label, target_h=h, target_w=w)
            resized_label = resized_label[:, :, 0].astype(dtype=np.uint8)
            resized_label = np.eye(self.num_classes())[resized_label]  # convert to one hot (h, w, c)

        else:
            raise NotImplementedError('unknown resize mode {}'.format(resize_mode))
        assert resized_image.shape[:2] == resized_label.shape[:2]
        assert len(resized_image.shape) == 3
        assert len(resized_label.shape) == 3

        if flatten_labels:
            data_shape = resized_label.shape[0] * resized_label.shape[1]
            nc = resized_label.shape[2]
            resized_label = np.reshape(resized_label, (data_shape, nc))
            assert len(resized_label.shape) == 2

        return resized_image, resized_label

    def flow(self, transform=True, batch=True, single_pass=False, **kwargs):
        def naive_flow():
            if self._config.instance_mode:
                return (self._retrieve_instance(item) for item in self._annotation_generator())
            elif self._config.merge_annotations:
                return self._combined_sample_generator()
            else:
                return (self._retrieve_sample(item) for item in self._annotation_generator())

        def secondary_flow():
            h = self._config.target_height
            w = self._config.target_width

            if batch:
                batch_size = self._config.batch_size
                target_images = np.zeros(shape=(batch_size, h, w, 3))
                target_labels = np.zeros(shape=(batch_size, h * w, self.num_classes()))
                for idx, (img, lbl) in enumerate(naive_flow()):
                    j = idx % batch_size
                    target_images[j], target_labels[j] = self.transform(img, lbl)
                    if j == batch_size - 1:
                        yield target_images, target_labels
            elif transform:
                for img, lbl in naive_flow():
                    yield self.transform(img, lbl)
            else:
                for img, lbl in naive_flow():
                    yield img, lbl

        while True:
            for image, label in secondary_flow():
                inputs = {'image': image}
                outputs = {'output': label}
                yield inputs, outputs
            if single_pass:
                break

    @staticmethod
    def mask_to_mscoco(alpha, annotations, img_id, mode='rle'):
        if mode == 'rle':
            in_ = np.reshape(np.asfortranarray(alpha), (alpha.shape[0], alpha.shape[1], 1))
            in_ = np.asfortranarray(in_)
            rle = mask.encode(in_)
            segmentation = rle[0]
        else:
            raise ValueError('Unknown mask mode "{}"'.format(mode))
        for idx, c in enumerate(np.unique(alpha)):
            area = mask.area(rle).tolist()
            if isinstance(area, list):
                area = area[0]
            bbox = mask.toBbox(rle).tolist()
            if isinstance(bbox[0], list):
                bbox = bbox[0]
            annotation = {
                'area': area,
                'bbox': bbox,
                'category_id': c,
                'id': len(annotations)+idx,
                'image_id': img_id,
                'iscrowd': 0,
                'segmentation': segmentation}
            annotations.append(annotation)
        return annotations

    def time(self):
        steps_per_epoch = 50

        start_time = time.clock()
        for idx, batch in enumerate(self.flow()):
            if idx + 1 >= steps_per_epoch:
                break
        elapsed = time.clock() - start_time

        print('Elapsed: {}, per sample: {}'.format(elapsed, elapsed/steps_per_epoch))


# class MSCOCOReduced(MSCOCO):
#     NAME = 'mscoco_reduced'
#
#     CATEGORIES = [
#         'background',  # class zero
#         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'baseball bat', 'bottle', 'wine glass',
#         'cup', 'fork', 'knife', 'spoon', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet',
#         'tv', 'laptop', 'cell phone', 'refrigerator', 'book', 'clock', 'vase']
#     IDS = [
#          0,
#          1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
#         11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#         22, 23, 27, 28, 31, 32, 33, 39, 44, 46,
#         47, 48, 49, 50, 61, 62, 63, 65, 67, 70,
#         72, 73, 77, 82, 84, 85, 86]
#
#     def __init__(self, config):
#         MSCOCO.__init__(self, config)
#
#         # pass through the dataset and count all valid samples
#         sample_counter = 0
#         imgs_new = {}
#         anns_new = {}
#         # cats_new = {0: {'id': 0, 'name': 'background', 'supercategory': 'background'}}
#         cats_new = {}
#         for cat_id in self.IDS[1:]:
#             cats_new[cat_id] = self._coco.cats[cat_id]
#         for img_id in self._coco.getImgIds():
#             annotation_ids = self._coco.getAnnIds(imgIds=img_id)
#             for annotation in self._coco.loadAnns(annotation_ids):
#                 if annotation['category_id'] not in MSCOCOReduced.IDS:
#                     continue
#                 if annotation['area'] > self._config.area_threshold:
#                     imgs_new[img_id] = self._coco.imgs[img_id]
#                     sample_counter += 1
#                     anns_new[annotation['id']] = self._coco.anns[annotation['id']]
#         self._coco.imgs = imgs_new
#         self._coco.anns = anns_new
#         self._coco.cats = cats_new
#         self._num_samples = sample_counter
#
#     def _annotation_generator(self, sample_size=None):
#         ann_generator = super(MSCOCOReduced, self)._annotation_generator(sample_size)
#         for ann in ann_generator:
#             if ann['category_id'] in MSCOCOReduced.IDS:
#                 yield ann


class DiskLoader(Dataset):
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.data_type = config['data_type']
        self.sample_size = config['sample_size']

    def collect_image_files_from_disk(self):
        """
        load file list from disk in pairs
        :return:
        """
        def join_paths(leaf):
            return os.path.join(self.data_dir, self.data_type, leaf)
        img_txt = join_paths('images.txt')
        lbl_txt = join_paths('labels.txt')
        with open(img_txt) as image_data, open(lbl_txt) as label_data:
            image_files = [join_paths(line.rstrip('\n')) for line in image_data]
            label_files = [join_paths(line.rstrip('\n')) for line in label_data]
            assert len(image_files) == len(label_files)
            files = zip(image_files, label_files)

        if self.sample_size:
            assert isinstance(self.sample_size, int)
            np.random.shuffle(files)
            files = files[:self.sample_size]
        return files

    def flow(self, transform=True, batch=True, single_pass=False,
             *args, **kwargs):
        for img_path, lbl_path in self.collect_image_files_from_disk():
            yield utils.load_image(img_path), utils.load_image(lbl_path)


mscoco = MSCOCO
disk = DiskLoader
