# from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict
from pycocotools.coco import COCO
import cv2
import os
import numpy as np


def palette():
    max_cid = max(ids()) + 1
    return [(cid, cid, cid) for cid in range(max_cid)]


def cids_to_ids_map():
    return {cid: idx for idx, cid in enumerate(ids())}


def ids():
    return [0,
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def id_to_palette_map():
    return {idx: color for idx, color in enumerate(palette())}
    # return {0: (0, 0, 0), idx: (idx, idx, idx) for idx, _ in enumerate(categories())}


def cid_to_palette_map():
    return {ids()[idx]: color for idx, color in enumerate(palette())}


def palette_to_id_map():
    return {color: ids()[idx] for idx, color in enumerate(palette())}
    # return {(0, 0, 0): 0, (idx, idx, idx): idx for idx, _ in enumerate(categories())}


def class_weighting():
    # weights = defaultdict(lambda: 1.5)
    weights = {i: 1.5 for i in ids()}
    weights[0] = 0.5
    return weights


def mask_to_palette_map(cid):
    mapper = id_to_palette_map()
    return {0: mapper[0], 255: mapper[cid]}


def categories():  # 80 classes
    return ['background',  # class zero
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def id_to_category(category_id):
    return {cid: categories()[idx] for idx, cid in enumerate(ids())}[category_id]


def category_to_cid_map():
    return {category: ids()[idx] for idx, category in enumerate(categories())}


def loadCOCO(data_dir, data_type):
    # dataset = 'data_mscoco/raw/annotations/instances_train2014.json'
    # dataDir='data_mscoco/raw/'
    # dataType='train2014'
    annFile='{}/annotations/instances_{}.json'.format(data_dir, data_type)
    coco = COCO(annFile)
    return coco


def yield_image(data_type, coco, target_shape=None):
    # anns = coco.annToMask()
    img_ids = coco.getImgIds()
    use_original_dims = not target_shape
    for idx, img_id in enumerate(img_ids):
        img = coco.loadImgs(img_id)[0]
        if use_original_dims:
            target_shape = (img['height'], img['width'], max(ids()) + 1)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask_one_hot = np.zeros(target_shape, dtype=np.uint8)
        mask_one_hot[:, :, 0] = 1  # every pixel begins as background
        # mask_one_hot = cv2.resize(mask_one_hot, target_shape[:2], interpolation=cv2.INTER_NEAREST)

        for ann in anns:
            mask_partial = coco.annToMask(ann)
            mask_partial = cv2.resize(mask_partial, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            assert mask_one_hot.shape[:2] == mask_partial.shape[:2]  # width and height match
#             print('another shape:', mask_one_hot[mask_partial > 0].shape)
            mask_one_hot[mask_partial > 0, ann['category_id']] = 1
            mask_one_hot[mask_partial > 0, 0] = 0
        yield img, mask_one_hot


def mask_to_mscoco(alpha, anns, img_id, mode='rle'):
    # mode can be poly or rle

    if mode == 'rle':  # not working (deepmask can't read it)
        in_ = np.reshape(np.asfortranarray(alpha), (alpha.shape[0], alpha.shape[1], 1))
        in_ = np.asfortranarray(in_)
        rle = mask.encode(in_)
        segmentation = rle[0]
    else:
        raise ValueError('Unknown mask mode "{}"'.format(mode))
    for idx, c in enumerate(np.distinct(alpha)):
        area = mask.area(rle).tolist()
        if isinstance(area, list):
            area = area[0]
        bbox = mask.toBbox(rle).tolist()
        if isinstance(bbox[0], list):
            bbox = bbox[0]
        ann = {'area': area,
               'bbox': bbox,
               'category_id': c,
               'id': len(anns)+idx,  
               'image_id': img_id,   
               'iscrowd': 0,
               'segmentation': segmentation}
        anns.append(ann)
    return anns
