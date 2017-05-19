from __future__ import print_function, division
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import sys
import json


def evaluate(eval_config):

    ######### read config file #########
    metadata = eval_config
    ann_type = metadata['ann_type']
    data_type = metadata['data_type']
    prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'
    model_name = metadata['model_name']
    # mscoco_dir = metadata['mscoco_dir']
    test_sample_size = metadata['test_sample_size'] if 'test_sample_size' in metadata else None
    ####################################

    ann_file = os.path.join('data', 'mscoco', 'annotations', '{}_{}.json'.format(prefix, data_type))

    evaluation_dir = os.path.join('models', 'mscoco', model_name, 'results')
    res_file = os.path.join(evaluation_dir, '{}_{}_{}_results.json'.format(prefix, data_type, ann_type))


    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_file)

    imgIds = sorted(coco_dt.getImgIds())
    test_sample_size = -1 if test_sample_size is None else test_sample_size
    imgIds = imgIds[:test_sample_size]

    # cocoEval = COCOeval(coco_gt, coco_dt, ann_type)
    cocoEval = COCOeval(coco_gt, coco_dt, ann_type)
    cocoEval.params.imgIds = imgIds
    # cocoEval.params.iouThrs = [i / 100 for i in range(5, 95, 30)] #[.05:.3:.95]
    # cocoEval.params.iouThrs = [0.05, 0.5]
    # cocoEval.params.iouThrs = [0.001]
    cocoEval.params.useCats = 0
    # cocoEval.params.maxDets = [1, 1, 1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_json = sys.argv[1]
    else:
        config_json = 'config/evaluation.json'
    config = json.load(open(config_json))
    evaluate(config)
