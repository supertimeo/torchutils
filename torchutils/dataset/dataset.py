from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import datasets, transforms
from PIL import Image
import os
import sys
import random
import numpy as np
import pandas as pd
import pycocotools
import pycocotools.mask as maskUtils
import pycocotools.coco as COCO
import pydantic


class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        """
        Args:
            img_dir (str): dossier des images
            ann_file (str): chemin vers le fichier json des annotations COCO
            transforms (callable, optional): transformations à appliquer à l'image et annotations
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Récupère l'id de l'image
        img_id = self.ids[index]
        # Récupère les infos de l'image (filename, height, width)
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        # Ouvre l'image
        img_path = os.path.join(self.img_dir, path)
        img = Image.open(img_path).convert("RGB")

        # Récupère les annotations pour cette image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prépare un dictionnaire des annotations
        target = {}
        # Par exemple, extraire les boîtes englobantes, labels, masques
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            boxes.append(ann['bbox'])  # bbox format COCO = [x,y,width,height]
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
            if 'segmentation' in ann:
                masks.append(ann['segmentation'])

        # Convertir en tenseurs PyTorch et formats attendus
        # Convertir bbox [x,y,w,h] en [x_min,y_min,x_max,y_max]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y + h

        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = areas
        target['iscrowd'] = iscrowd
        target['masks'] = masks  # ce sont des polygones COCO, pas encore des masks binaires

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target