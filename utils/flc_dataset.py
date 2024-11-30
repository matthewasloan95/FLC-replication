# let's create a pytorch dataset for the FLC dataset


import torch
from torch.utils.data import Dataset
# https://pytorch.org/vision/main/transforms.html this showcases what I believe to be the basic image transforms: normalization, toimg, resize... and should work with a dataloader with num_workers > 0
from torchvision.transforms import v2
# https://github.com/ppwwyyxx/cocoapi
from pycocotools.coco import COCO
# for loading images
import cv2
import numpy as np


class FLCDataset(Dataset):
    # so this dataset will need to take the dir of images, the annotation file, and the transform. It will also need to know if it's the train set or test set
    def __init__(self, root_dir,
                 annotation_file, 
                 transform=None, 
                 train=False):
        
        # this debug flag will be used to make sure what I'm doing is correct and lining up
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.transform = transform
            
        # now we need to make sure the annotations have matching ids and images
        ids_with_anns = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                ids_with_anns.append(img_id)
        self.ids = ids_with_anns
        
        self.train = train

    # len
    def __len__(self):
        return len(self.ids)
    
    # getitem
    def __getitem__(self, idx):
        # the goal here is to return the image, list of labels, list of bbox's, masks, and image_id
        # I think target will be a dict with all of these except image
         
        # let's remember that coco is 1-indexed and contains this data in the id's: train_dataset[0] - > [{'id': 1, 'file_name': '1_000001.jpg', 'width': 4128, 'height': 2322,...}]
        image_id = self.ids[idx]
        image_data = self.coco.loadImgs(image_id)[0]
        
        # load the image
        image = cv2.imread(f'{self.root_dir}/{image_data["file_name"]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)
            
        # ok now each image has a list of annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # each annotation has:
        # id, image_id, category_id, iscrowd, area, bbox, segmentation, width, height
        
                
        # now we know we have the format down, let's create the target dict
        boxes = []
        labels = []
        masks = []
        for ann in anns:
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x+w, y+h]) # x, y, width, height where xy is upper left corner
            labels.append(ann['category_id'])
            mask = self.coco.annToMask(ann) # glad coco has this function
            masks.append(mask)
            
        # we're working with tensors
        #TODO: check out torchvision.tv_tensors.BoundingBoxes and torchvision.tv_tensors.Masks
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id])
        }
        
        for k, v in target.items():
            # if v is not a tensor
            if not torch.is_tensor(v):
                target[k] = torch.tensor(v)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = image/255.0
        
        return image, target
