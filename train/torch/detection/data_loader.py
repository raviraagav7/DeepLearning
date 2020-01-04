import os
import numpy as np
import torch
from labelbox import Client
from PIL import Image
import sys
from torch.utils.data import Dataset
import wget
import json
import ssl
import glob
import argparse
import random
sys.path.append('.')
from data_aug.data_aug import *
import utils.transforms as T
from settings import LABELBOX_API_KEY

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class SOW2Dataset(Dataset):

    def __init__(self, project_name, image_dir, transforms, mode='train'):
        self.client = Client()
        self.mode = mode
        self.project_name = project_name
        self.image_dir = image_dir
        self.transforms = transforms
        self.label_dict = None
        self.image_dict = dict()
        self.unique_label = set()
        self.parse_json()
        self.unique_labels()
        self.classes = sorted(list(set([i.split(' ')[0] for i in self.unique_label])))
        # self.classes = ['Anode', 'Debris', 'Boulder']
        self.load_folder_images()


    def export_labels(self):
        """

        This function exports the labels for the specified LabelBox project.

        Returns:
            None

        """
        flag = False
        for project in self.client.get_projects():
            if project.name != self.project_name:
                continue
            else:
                flag = True
                break
        if flag:
            project = self.client.get_project(project_id=str(project.uid))
            url = project.export_labels()
            print(url)
            wget.download(url, './labels.json')
        else:
            raise Exception("Project mentioned does not exist in LabelBox.")

    def parse_json(self):
        """

        This function parse the exported JSON file.

        Returns:
            None

        """
        if not os.path.isfile(os.path.join(os.path.dirname(__file__), 'labels.json')):
            self.export_labels()
        with open(os.path.join(os.path.dirname(__file__), 'labels.json'), 'r') as f_read:
            self.label_dict = json.load(f_read)

        self.label_dict = list(filter(lambda i: i['Label'] != 'Skip', self.label_dict))
        self.label_dict = list(filter(lambda i: type(i['Label']) is not str, self.label_dict))
        self.label_dict = list(filter(lambda i: len(i['Label']) is not 0, self.label_dict))

    def load_folder_images(self):
        """

        This function creates a dictionary of Image Name and its corresponding File Path.

        Returns:
            None

        """
        for file_path in glob.glob(os.path.join(self.image_dir, '*.jpg')):
            self.image_dict[file_path.split('/')[-1]] = file_path

    def unique_labels(self):
        """

        This function finds the unique labels in the entire list of Labels.

        Returns:
            None

        """
        for label in self.label_dict:
            if label['Label'] == 'Skip':
                continue
            for key in label['Label'].keys():
                self.unique_label.add(key)

    def __getitem__(self, idx):

        image_name = list(self.image_dict.keys())[idx]
        image_path = self.image_dict[image_name]
        label_data = list(filter(lambda i: image_name in i['Labeled Data'], self.label_dict))[0]
        img = Image.open(image_path).convert("RGB")

        obj_ids = label_data['Label'].keys()

        bboxes = []
        # boxes = []
        # labels = []
        for id in obj_ids:
            for box in label_data['Label'][id]:
                if box['geometry'][0]['x'] < box['geometry'][2]['x']:
                    x_min = box['geometry'][0]['x']
                    x_max = box['geometry'][2]['x']
                else:
                    x_min = box['geometry'][2]['x']
                    x_max = box['geometry'][0]['x']

                if box['geometry'][0]['y'] < box['geometry'][2]['y']:
                    y_min = box['geometry'][0]['y']
                    y_max = box['geometry'][2]['y']
                else:
                    y_min = box['geometry'][2]['y']
                    y_max = box['geometry'][0]['y']
                bboxes.append([x_min, y_min, x_max, y_max, self.classes.index(id.split(' ')[0]) + 1])
                # boxes.append([x_min, y_min, x_max, y_max])
                # labels.append(self.classes.index(id.split(' ')[0]) + 1)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if self.mode == 'train':
            img = np.asarray(img)
            img, bboxes = RandomHorizontalFlip(1)(img, bboxes)
            # print(bboxes)
            img = Image.fromarray(img)

        boxes = bboxes[:, :4]
        labels = bboxes[:, -1]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except Exception:
            print("error")
        # suppose all instances are not crowd
        is_crowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = is_crowd
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the LabelBox labels.')
    parser.add_argument('-p', '--project_name', nargs='?', type=str, default='Underwater Project',
                        help='Project Name in LabelBox for downloading the labels')
    parser.add_argument('-d', '--image_dir', nargs='?', type=str,
                        default='/home/ravi/dataset/batch-1/train',
                        help='Required if the url_image (-u) argument is not used.')

    args = parser.parse_args()
    dataset = SOW2Dataset(args.project_name, args.image_dir, get_transform(train=True), mode='test')
    dataset[0]