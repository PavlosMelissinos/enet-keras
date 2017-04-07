from __future__ import print_function, division, absolute_import
import sys
import mscoco
import os
import cv2
from utils import one_hot_to_rgb


if __name__ == "__main__":
    data_dir = sys.argv[1]
    data_type = sys.argv[2]
    target_dir = sys.argv[3] if len(sys.argv) > 3 else None
    if data_type not in ['train2014', 'val2014']:
        raise ValueError('Data type {} not recognized'.format(data_type))
    target_dir = data_dir if not target_dir else target_dir
    coco = mscoco.loadCOCO(data_dir, data_type)

    total = len(coco.imgs)
    idx = 0
    for res in mscoco.yield_image(coco):
        if not res:
            status = 'Skip'
        else:
            img, mask = res[0], res[1]
            # id_to_palette = mscoco.id_to_palette_map()
            rgb_label = one_hot_to_rgb(mask)
            filename_no_ext = os.path.splitext(img['file_name'])[0]
            lbl_path = os.path.join(target_dir, data_type, 'labels', '{}.png'.format(filename_no_ext))
            cv2.imwrite(lbl_path, rgb_label)  # TODO: should this be reversed (rgb -> bgr)? ditch cv2
            status = 'OK'
        idx += 1
        print('Processed {} out of {} images. Status: {}'.format(idx, total, status), end='\r')
        sys.stdout.flush()
