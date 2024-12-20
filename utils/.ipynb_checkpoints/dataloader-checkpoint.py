import torch, os
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset
from PIL import Image 
import pandas as pd
from .parse_xml import parseXML, parseXML_pascal, get_label
from .config import *

'''
Classification loader are quite obsolete as they were used for non-part-based classification (such as resnet) and to train baseline
'''


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
val_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

class PascalClassificationDataset(object):
    def __init__(self, relevant_xml, img_folder, xml_folder, transform=None, batch_size=8):
        rel_xml = open(relevant_xml, 'r')
        list_xml = [line.split('\n')[0] for line in rel_xml.readlines()]
        self.images_loc = [os.path.join(img_folder,path[:-4]+'.jpg') for path in list_xml]
        self.xml_loc = [os.path.join(xml_folder,path) for path in list_xml]
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images_loc)//self.batch_size + (len(self.images_loc)%self.batch_size > 0)

    def __getitem__(self, idx):

        imgs = []
        labels = []
        
        for k in range(self.batch_size*idx, min(self.batch_size*(idx+1), len(self.images_loc))):
            img_name = self.images_loc[k]
            image = Image.open(img_name).convert('RGB')
            labels.append(get_label(self.xml_loc[k], PASCAL_EL_DIC))
            
            if self.transform:
                imgs.append(self.transform(image))
            else:
                imgs.append(np.asarray(image))

        return torch.stack(imgs), torch.LongTensor(labels)

    def __len__(self):
        return len(self.images_loc)

class ArchitectureClassificationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, batch_size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_loc = pd.read_csv(csv_file)
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images_loc)//self.batch_size + (len(self.images_loc)%self.batch_size > 0)

    def __getitem__(self, idx):

        imgs = []
        labels = []
        
        for k in range(self.batch_size*idx, min(self.batch_size*(idx+1), len(self.images_loc))):

            img_name = self.images_loc.iloc[k, 0]
            image = Image.open(img_name).convert('RGB')
            labels.append(self.images_loc.iloc[k, 1])
            if self.transform:
                imgs.append(self.transform(image))

        return torch.stack(imgs), torch.LongTensor(labels)


from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    Normalize
)
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
import numpy as np 
from skimage.transform import resize 
transform_detection =  Compose(
    [
        Resize(256,256),
        CenterCrop(224,224),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensor()
    ],
    bbox_params={'format':'pascal_voc', 'label_fields':['labels']}, 
    p=1
                                )
                            

transform_detection_pascal =  Compose(
    [
        Resize(224,224),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensor()
    ],
    bbox_params={'format':'pascal_voc', 'label_fields':['labels']}, 
    p=1
                                )

transform_detection_eval =  T.Compose([T.Resize(256), T.CenterCrop(224)])

def compute_area(list_boxes):
    return np.array([(box[2] - box[0])*(box[3]-box[1]) for box in list_boxes])

class ArchitectureDetectionDataset(object):
    def __init__(self, csv_img, csv_xml, transform=None, batch_size=1):
        self.images_loc = pd.read_csv(csv_img)
        self.xml_loc = pd.read_csv(csv_xml)
        self.transform = transform
        self.batch_size = batch_size

    def __getitem__(self, idx):

        img_name = self.images_loc.iloc[idx, 0]
        image = np.asarray(Image.open(img_name).convert('RGB'))
        im_name = img_name.split('/')[-1][:-4]
        xml_name, label = self.xml_loc[self.xml_loc['path'].str.contains(im_name)].values[0]
        name, folder, shape, target = parseXML(xml_name, SUB_ELEMENTS, idx, False)
        image = resize(image,shape,clip=False,preserve_range=True)
        if self.transform:
            augmented = self.transform(image=image, bboxes=target['boxes'], labels=target['labels'])
            img, target['boxes'] = augmented['image'], np.array(augmented['bboxes'])
            target['area'] = compute_area(target['boxes'])
        target['boxes'] = torch.from_numpy(target['boxes']).type(torch.FloatTensor)
        target['area'] = torch.from_numpy(target['area']).type(torch.FloatTensor)
        target['labels'] = torch.from_numpy(target['labels']).type(torch.int64)
        target['image_id'] = torch.from_numpy(target['image_id']).type(torch.int64)
        target['iscrowd'] = torch.from_numpy(target['iscrowd']).type(torch.uint8)

        return [img], [target]

    def __len__(self):
        return len(self.images_loc)


class PascalDetectionDataset(object):
    def __init__(self, relevant_xml, img_folder, xml_folder, transform=None, batch_size=1):
        rel_xml = open(relevant_xml, 'r')
        list_xml = [line.split('\n')[0] for line in rel_xml.readlines()]
        self.images_loc = [os.path.join(img_folder,path[:-4]+'.jpg') for path in list_xml]
        self.xml_loc = [os.path.join(xml_folder,path) for path in list_xml]
        self.transform = transform
        self.batch_size = batch_size

    def __getitem__(self, idx):
        img_name = self.images_loc[idx]
        image = np.asarray(Image.open(img_name).convert('RGB'))
        im_name = img_name.split('/')[-1][:-4]
        xml_name = self.xml_loc[idx]
        name, folder, shape, target = parseXML_pascal(xml_name, PASCAL_PART_DIC, idx, False)
        image = resize(image,shape,clip=False,preserve_range=True)
        #print(target['boxes'])
        #print(image.shape)
        if self.transform:
            augmented = self.transform(image=image, bboxes=target['boxes'], labels=target['labels'])
            img, target['boxes'] = augmented['image'], np.array(augmented['bboxes'])
            target['area'] = compute_area(target['boxes'])
        #print(target['boxes'])
        target['boxes'] = torch.from_numpy(target['boxes']).type(torch.FloatTensor)
        target['area'] = torch.from_numpy(target['area']).type(torch.FloatTensor)
        target['labels'] = torch.from_numpy(target['labels']).type(torch.int64)
        target['image_id'] = torch.from_numpy(target['image_id']).type(torch.int64)
        target['iscrowd'] = torch.from_numpy(target['iscrowd']).type(torch.uint8)
        #print(target['boxes'])
        #print('----------------')

        return [img], [target]

    def __len__(self):
        return len(self.images_loc)