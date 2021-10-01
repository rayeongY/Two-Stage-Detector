import os
import numpy as np

import torch
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
            

def anchor_generator(batch_size, w, h, sub_smpl):
    
    f_w = w // sub_smpl
    f_h = h // sub_smpl
    full_anchor_map = torch.zeros([batch_size, f_w, f_h, 9, 4])
    cell_x_offset = torch.arange(0, f_w, 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat((1, f_h, 1, 9)).unsqueeze(0)
    cell_y_offset = torch.arange(0, f_h, 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat((1, f_w, 1, 9)).unsqueeze(-1)

    scales = torch.tensor([sub_smpl // 2, sub_smpl, sub_smpl * 2])
    scales = scales.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 2)
    
    ratios = torch.tensor([[np.sqrt(0.5),   np.sqrt(2)],
                           [  np.sqrt(1),   np.sqrt(1)],
                           [  np.sqrt(2), np.sqrt(0.5)]]).unsqueeze(0)
    anchors = sub_smpl * scales * ratios
    anchors = anchors.reshape(-1, 2).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    full_anchor_map[..., 0] = cell_x_offset
    full_anchor_map[..., 1] = cell_y_offset
    full_anchor_map[..., 2:] = anchors.repeat(1, f_w, f_h, 1, 1)

    ## if non-xywh format, need trans to xyxy format

    return full_anchor_map


def t_anchor_generator():
    pass


def t_prop_generator():
    pass


def nms():
    pass