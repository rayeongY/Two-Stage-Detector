import os
import numpy as np

from PIL import Image
from pycocotools.coco import COCO

import torch
import torchvision
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(
        self,
        dataset_opt,
        model_opt,
        split="train",
    ):
        super(DataSet, self).__init__()

        self.daodataset_opt =dataset_opt
        self.model_opt = model_opt
        self.classes = self.daodataset_opt["DATASET"]["CLASSES"]
        
        dataset_name =dataset_opt["DATASET"]["NAME"]

        assert split == "train" or split == "valid"
        # assert dataset_name in ["ship", "yolo-dataset"]
                
        # if dataset_name == "yolo-dataset" or dataset_name == "ship":
        if split == "train":
            dataset_type = "train"
        elif split == "valid":
            dataset_type = "valid"

        # root = self.daodataset_opt["DATASET"]["ROOT"]
        self.split = split
        
        # self.dataset = self.load_dataset(os.path.join(root, dataset_type))
        self.dataset = self.load_dataset(self.split)


    def __getitem__(self, idx):
        
        img_path, label_path = self.dataset[idx]

        ## load img
        img_file = Image.open(img_path)
        t = torchvision.transforms.Compose([torchvision.transforms.Resize((800, 800)), torchvision.transforms.ToTensor()])

        img_file = t(img_file)

        # ## load label
        # label_f = open(label_path, "r")

        # labels = np.zeros((0, 5))
        # if os.fstat(label_f.fileno()).st_size:
        #     labels = np.loadtxt(label_f, dtype="float")
        #     labels = labels.reshape(-1, 5)

        # label_maps = create_label_map(labels, self.model_opt)

        return img_file, labels, img_path


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        pass


        _data = {
            "img": ...,
            # "label": ...,
            "img_path": ...,
        }
        return _data


    # def load_dataset(self, f_list_path):
    def load_dataset(self, dataset_path):
        
        coco = COCO(dataset_path)

        ## 




        image_set = []

        # for r, _, f in os.walk(dataset_path):
        #     for file in f:
        #         if file.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
        #             # set paths - both image and label file
        #             img_path = os.path.join(r, file).replace(os.sep, '/')
        #             label_path = os.path.splitext(img_path)[0] + ".txt"

        #             if not os.path.isfile(img_path) or not os.path.isfile(label_path):
        #                 continue
                                
        #             image_set.append((img_path, label_path))
                
        return image_set
    

    def collate_fn(self, batch):
        batch_input = {}

        img_files = []
        # labels = []
        img_paths = []
        # maps_0 = []
        # maps_1 = []
        # maps_2 = []

        for b in batch:
            img_files.append(b[0])
        #     labels.append(b[1])
            img_paths.append(b[2])

        #     maps_0.append(b[3][0])
        #     maps_1.append(b[3][1])
        #     maps_2.append(b[3][2])

        img_files = torch.stack(img_files, 0)

        # maps_0 = torch.stack(maps_0, 0)
        # maps_1 = torch.stack(maps_1, 0)
        # maps_2 = torch.stack(maps_2, 0)
        # label_maps = [maps_0, maps_1, maps_2]

        batch_input = {
            "img": img_files,
        #     "label": labels,
            "img_path": img_paths,
        #     "label_map": label_maps
        }

        return batch_input


