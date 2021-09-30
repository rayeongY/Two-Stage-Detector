from common.utils import create_label_files

import os
import numpy as np

from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(
        self,
        dataset_opt,
        model_opt,
        split="train",
        json_path=None
    ):
        super(DataSet, self).__init__()

        self.daodataset_opt =dataset_opt
        self.model_opt = model_opt
        self.classes = self.daodataset_opt["DATASET"]["CLASSES"]
        
        dataset_name = dataset_opt["DATASET"]["NAME"]

        assert split == "train" or split == "valid"
        # assert dataset_name in ["ship", "yolo-dataset"]
                
        # if dataset_name == "yolo-dataset" or dataset_name == "ship":
        if split == "train":
            self.split = "train"
        elif split == "valid":
            self.split = "valid"

        if not json_path is None:
            create_label_files(dataset_opt, json_path)

        self.dataset = self.load_dataset()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        img_path, label_path = self.dataset[index]

        ## load img
        img_file = Image.open(img_path)
        t = torchvision.transforms.Compose([torchvision.transforms.Resize((496, 496)), torchvision.transforms.ToTensor()])

        img_file = t(img_file)

        ## load label
        label_f = open(label_path, "r")

        labels = np.zeros((0, 5))
        if os.fstat(label_f.fileno()).st_size:
            labels = np.loadtxt(label_f, dtype="float")
            labels = labels.reshape(-1, 5)

        # label_maps = create_label_map(labels, self.model_opt)

        _data = {
            "img": img_file,
            "label": labels,
            "img_path": img_path,
            # "label_map": label_maps
        }
        return _data


    # def load_dataset(self, f_list_path):
    def load_dataset(self):
        root = self.dataset_opt["DATASET"]["ROOT"]
        dataset_path = os.path.join(root, self.split).replace(os.sep, "/")


        image_set = []

        for r, _, f in os.walk(dataset_path):
            for file in f:
                if file.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
                    # set paths - both image and label file
                    img_path = os.path.join(r, file).replace(os.sep, '/')
                    label_path = os.path.splitext(img_path)[0] + ".txt"

                    if not os.path.isfile(img_path) or not os.path.isfile(label_path):
                        continue
                                
                    image_set.append((img_path, label_path))
                
        return image_set
    

    def collate_fn(self, batch):
        batch_input = {}

        img_files = []
        labels = []
        img_paths = []
        # label_maps = []

        for b in batch:
            img_files.append(b["img"])
            labels.append(b["label"])
            img_paths.append(b["img_path"])
            # label_maps.append(b["label_map"])

        img_files = torch.stack(img_files, 0)
        # labels = torch.stack(labels, 0)
        # label_maps = torch.stack(label_maps, 0)

        batch_input = {
            "img": img_files,
            "label": labels,
            "img_path": img_paths,
        #     "label_map": label_maps
        }

        return batch_input


