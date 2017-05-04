# coding=utf-8
from __future__ import absolute_import, division, print_function
from pycocotools import mask
from pycocotools.coco import COCO
import numpy as np
import numbers
import abc
import os
from . import utils


def load(dataset_name, data_dir=None, data_type=None):
    if dataset_name == 'mscoco':
        if data_dir is None or data_type is None:
            return MSCOCO
        else:
            return MSCOCO(data_dir=data_dir, data_type=data_type)
    else:
        raise NotImplementedError('Unknown dataset {}'.format('dataset_name'))


class Dataset(object):
    __metaclass__ = abc.ABCMeta

    NAME = 'dataset'
    CATEGORIES = []
    IDS = []
    PALETTE = []

    # def __init(self, data_dir, data_type):
    #     # this function should be overriden
    #     pass

    @abc.abstractmethod
    def load(self, data_dir, data_type):
        """Method documentation"""
    #
    # def id_to_category(self, primary_id):
    #     return self.CATEGORIES[primary_id]


class MSCOCO(Dataset):
    NAME = 'mscoco'
    CATEGORIES = [
        'background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
    IDS = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    PALETTE = [(cid, cid, cid) for cid in range(max(IDS) + 1)]

    def __init__(self, data_dir, data_type):
        """
        
        :param data_dir: base ms-coco path (parent of annotation json directory)
        :param data_type: 'train2014', 'val2014' or 'test2015'
        """
        Dataset.__init__(self)

        valid_data_types = ['train2014', 'val2014', 'test2015']
        if data_type not in valid_data_types:
            raise ValueError('Unknown data type {}. Valid values are {}'.format(data_type, valid_data_types))

        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        print('Initializing MS-COCO: Loading annotations from {}'.format(annotation_file))
        self._coco = COCO(annotation_file)

        # self._data_dir = os.path.join(data_dir, data_type)
        self._data_dir = {'annotations': annotation_file,
                          'images': os.path.join(data_dir, data_type, 'images')}

        self._area_threshold = 2500

        # build indices
        self._cid_to_id = {cid: idx for idx, cid in enumerate(self.IDS)}
        self._category_to_id = {category: idx for idx, category in enumerate(self.CATEGORIES)}
        self._palette_to_id = {category: idx for idx, category in enumerate(self.CATEGORIES)}

    @property
    def categories(self):
        return MSCOCO.CATEGORIES

    @property
    def palette(self):
        return MSCOCO.PALETTE

    @property
    def size(self):
        return len(self._coco.imgs)

    @staticmethod
    def num_classes():
        return len(MSCOCO.IDS)

    def load(self, data_dir, data_type):
        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        print('Initializing MS-COCO: Loading annotations from {}'.format(annotation_file))
        self._coco = COCO(annotation_file)

    def _annotation_generator(self, sample_size=None):
        """
        Generates sample_size annotations. No pre/post-processing
        :type sample_size: int
        :param sample_size: How many samples to retrieve before stopping
        :return: coco annotation (dictionary)
        """

        img_ids = self._coco.getImgIds()
        if isinstance(sample_size, numbers.Number):
            # TODO: random.sample might cause problems when target size is larger than image dimensions \
            # TODO: Examine shuffle + break as an alternative
            img_ids = np.random.choice(img_ids, size=int(sample_size))

        for idx, img_id in enumerate(img_ids):
            annotation_ids = self._coco.getAnnIds(imgIds=img_id)
            for annotation in self._coco.loadAnns(annotation_ids):
                if annotation['area'] > self._area_threshold:
                    yield annotation

    def _crop_detection(self, annotation):
        """
        crops a pair of image label arrays according to an annotation id
        :type annotation: dict
        """

        x, y, width, height = [int(elem + 0.5) for elem in annotation['bbox']]

        coco_image = self._coco.loadImgs(annotation['image_id'])[0]
        image_path = os.path.join(self._data_dir['images'], coco_image['file_name'])
        image = utils.load_image(image_path)

        ann_mask = self._coco.annToMask(annotation)

        mask_categorical = np.zeros((ann_mask.shape[0], ann_mask.shape[1], self.num_classes()), dtype=np.uint8)
        mask_categorical[:, :, 0] = 1  # every pixel begins as background

        class_index = self._cid_to_id[annotation['category_id']]
        mask_categorical[ann_mask > 0, class_index] = 1
        mask_categorical[ann_mask > 0, 0] = 0  # remove background label from pixels of this (non-bg) category

        cropped_image = image[y: y + height, x: x + width, :]
        cropped_label = mask_categorical[y: y + height, x: x + width, :]
        return cropped_image, cropped_label

    def crop_generator(self, sample_size=None):
        for ann in self._annotation_generator(sample_size=sample_size):
            cropped_image, cropped_label = self._crop_detection(ann)
            yield cropped_image, cropped_label

    def sample_generator(self, sample_size=None):
        """
        Generates image/mask pairs from dataset.
        MS-COCO categories have some gaps among the ids. This function compresses them to eliminate these gaps.
        :type sample_size: int
        :param cover_gaps: if True, the category ids are compressed so that gaps are eliminated (useful for training)
        :param sample_size: How many samples to retrieve before stopping
        :return: generator of image-mask pairs.
        Each pair is a tuple. Image shape is (height, width, 3) and the shape of the mask is (height, width, 91)
        """

        for annotation in self._annotation_generator(sample_size=sample_size):
            coco_image = self._coco.loadImgs(annotation['image_id'])[0]
            image_path = os.path.join(self._data_dir['images'], coco_image['file_name'])
            image = utils.load_image(image_path)

            ann_mask = self._coco.annToMask(annotation)

            mask_categorical = np.zeros((ann_mask.shape[0], ann_mask.shape[1], self.num_classes()), dtype=np.uint8)
            mask_categorical[:, :, 0] = 1  # every pixel begins as background

            class_index = self._cid_to_id[annotation['category_id']]
            mask_categorical[ann_mask > 0, class_index] = 1
            mask_categorical[ann_mask > 0, 0] = 0  # remove background label from pixels of this (non-bg) category
            yield image, mask_categorical

    def combined_sample_generator(self, cover_gaps=True, target_h=None, target_w=None, sample_size=None):
        """
        Generates image/mask pairs from dataset.
        MS-COCO categories have some gaps among the ids. This function compresses them to eliminate these gaps.
        :type sample_size: int
        :param cover_gaps: if True, the category ids are compressed so that gaps are eliminated (useful for training)
        :param target_h:
        :param target_w:
        :param sample_size:
        :return: generator of image-mask pairs.
        Each pair is a tuple. Image shape is (width, height, 3) and the shape of the mask is (width, height, 91)
        """
        default_target_h = target_h
        default_target_w = target_w

        img_ids = self._coco.getImgIds()
        if isinstance(sample_size, numbers.Number):
            # TODO: random.sample might cause problems when target size is larger than image dimensions \
            # TODO: Examine shuffle + break as an alternative
            img_ids = np.random.choice(img_ids, size=sample_size)

        for idx, img_id in enumerate(img_ids):
            coco_image = self._coco.loadImgs(img_id)[0]

            image_path = os.path.join(self._data_dir['images'], coco_image['file_name'])
            image = utils.load_image(image_path)

            dimensions = len(self.IDS) if cover_gaps else max(self.IDS) + 1
            target_h = coco_image['height'] if not default_target_h else default_target_h
            target_w = coco_image['width'] if not default_target_w else default_target_w

            if target_h > coco_image['height'] or target_w > coco_image['width']:
                continue

            image = utils.resize(image, target_h=target_h, target_w=target_w)

            mask_one_hot = np.zeros((target_h, target_w, dimensions), dtype=np.uint8)
            mask_one_hot[:, :, 0] = 1  # every pixel begins as background

            annotation_ids = self._coco.getAnnIds(imgIds=coco_image['id'])

            for annotation in self._coco.loadAnns(annotation_ids):
                mask_partial = self._coco.annToMask(annotation)
                mask_partial = utils.resize(mask_partial, target_h=target_h, target_w=target_w)
                assert mask_one_hot.shape[:2] == mask_partial.shape[:2]  # width and height match
                if cover_gaps:
                    class_index = self._cid_to_id[annotation['category_id']]
                else:
                    class_index = annotation['category_id']
                assert class_index > 0
                mask_one_hot[mask_partial > 0, class_index] = 1
                mask_one_hot[mask_partial > 0, 0] = 0  # remove background label from pixels of this (non-bg) category
            yield image, mask_one_hot

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
