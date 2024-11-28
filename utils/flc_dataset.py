# let's create a pytorch dataset for the FLC dataset


import torch
from torch.utils.data import Dataset
# https://pytorch.org/vision/main/transforms.html this showcases what I believe to be the basic image transforms: normalization, toimg, resize... and should work with a dataloader with num_workers > 0
from torchvision.transforms import v2
# https://github.com/ppwwyyxx/cocoapi
from pycocotools.coco import COCO
# for loading images
import cv2
# for quickly showing the image
import matplotlib.pyplot as plt


class FLCDataset(Dataset):
    # so this dataset will need to take the dir of images, the annotation file, and the transform. It will also need to know if it's the train set or test set
    def __init__(self, root_dir,
                 annotation_file, 
                 transform=None, 
                 train=False,
                 debug=False):
        
        # this debug flag will be used to make sure what I'm doing is correct and lining up
        self.debug = debug
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # if transform is none we'll do the basic transforms
        if transform is None:
            # NOTE: if I end up resizing I should consider going to uint8 -> resize -> float32 and compare the speeds
            self.transform = v2.Compose([
                # this ToImage and ToDtype is the equivalent of the ToTensor transform
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # now we need to make sure the annotations have matching ids and images
        ids_with_anns = set(self.coco.getAnnIds(self.ids))
        for img_id in self.ids:
            if img_id not in ids_with_anns:
                self.ids.remove(img_id)
        self.ids = list(sorted(self.ids))
        
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
        # display the image inline
        if self.debug:
            print(f'idx: {idx} | image_data: {image_data}')
            plt.imshow(image)
            
        # ok now each image has a list of annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # each annotation has:
        # id, image_id, category_id, iscrowd, area, bbox, segmentation, width, height
        
        if self.debug:
            print(f'anns: {anns}') 
            print(f'len(anns): {len} | len(anns[0]): {len(anns[0])}')
            print(f'ann_ids: {ann_ids}')
            print(f'len(ann_ids): {len(ann_ids)}')
            print(f'keys:')
            for ann in anns:
                for key in ann.keys():
                    print(f'{key}: {ann[key]}')
                    
            # show image with bbox's in plt, red for four leaf, blue for three leaf
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                color = 'r' if ann['category_id'] == 1 else 'b'
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
                # add the label in white with black border
                plt.text(x, y, f'{ann["category_id"]}', color='white', bbox=dict(facecolor='black', alpha=0.5))
                
            # show image with masks in plt, purple for four leaf, black for three leaf
            for ann in anns:
                mask = self.coco.annToMask(ann)
                color = 'purple' if ann['category_id'] == 1 else 'black'
                plt.imshow(mask, alpha=0.5)#, alpha=0.33, cmap='gray')
                
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
            
        
        plt.show()
        return 1
