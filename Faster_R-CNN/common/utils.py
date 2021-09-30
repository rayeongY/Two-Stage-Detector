import os
import numpy as np

from pycocotools.coco import COCO


def create_label_files(dataset_opt, json_path):
    '''
    This function tries to translate coco-format .json annotation files to .txt label files
    in the path where the related image files are located.
    
    created label file example:
        (if the image file is "00000009.jpg")
        - file name: "00000009.txt"
        - inner contents(format: obj_id x, y, w, h): 428.900 215.340 76.000 91.657 3.000
    '''
    coco = COCO(json_path)
    imgs = coco.dataset['images']

    root = dataset_opt["DATASET"]["ROOT"]
    for img in imgs:
        _file_path = root + img['coco_url'].split('/')[-2:]
        _id = img['id']

        labels = []
        anns = coco.loadAnns(_id)
        for ann in anns:
            label = ann['bbox']
            label.append(ann['category_id'])

            labels.append(label)

        labels = np.array(label)
        label_path = _file_path.split(".")[0] + ".txt"
        np.savetxt(label_path, label)
            

def anchor_generator():
    pass


def t_anchor_generator():
    pass


def t_prop_generator():
    pass


def nms():
    pass