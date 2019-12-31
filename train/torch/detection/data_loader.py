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
sys.path.append('.')
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

    def __init__(self, project_name, url_image, image_dir, transforms):
        self.client = Client()
        self.project_name = project_name
        self.url_image = url_image
        self.image_dir = image_dir
        self.transforms = transforms
        self.label_dict = None
        self.image_dict = dict()
        self.unique_label = set()
        self.parse_json()
        self.unique_labels()
        self.classes = list(set([i.split(' ')[0] for i in self.unique_label]))
        if not self.url_image and self.image_dir:
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
        self.export_labels()
        with open('./labels.json', 'r') as f_read:
            self.label_dict = json.load(f_read)

        self.label_dict = list(filter(lambda i: i['Label'] != 'Skip', self.label_dict))

    def load_folder_images(self):
        """

        This function creates a dictionary of Image Name and its corresponding File Path.

        Returns:
            None

        """
        list_dir = os.listdir(self.image_dir)
        for directory in list_dir:
            if directory is '.DS_Store' or (not os.path.isdir(os.path.join(self.image_dir, directory))):
                continue
            for file_path in glob.glob(os.path.join(self.image_dir, directory, '*.jpg')):
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

        url = self.label_dict[idx]['Labeled Data']
        image_name = '.'.join([url.split('/')[-1].split('.')[0], 'jpg'])
        img = Image.open(self.image_dict[image_name]).convert("RGB")

        obj_ids = self.label_dict[idx]['Label'].keys()

        boxes = []
        labels = []
        for id in obj_ids:
            for box in self.label_dict[idx]['Label'][id]:
                boxes.append([box['geometry'][0]['x'], box['geometry'][0]['y'],
                              box['geometry'][2]['x'], box['geometry'][2]['y']])
                labels.append(self.classes.index(id.split(' ')[0]))
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
        return len(self.label_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the LabelBox labels.')
    parser.add_argument('-p', '--project_name', nargs='?', type=str, default='Underwater Project',
                        help='Project Name in LabelBox for downloading the labels')
    parser.add_argument('-u', '--url_image', action='store_true',
                        help='Whether to download from the specified url (-u)')
    parser.add_argument('-d', '--image_dir', nargs='?', type=str,
                        default='/Users/srinivasraviraagav/Kespry-Dataset/sow-2-data/batch-1',
                        help='Required if the url_image (-u) argument is not used.')

    args = parser.parse_args()
    if not args.url_image and args.image_dir is None:
        raise Exception('Please provide the path to the Image Directory.')
    dataset = SOW2Dataset(args.project_name, args.url_image, args.image_dir, get_transform(train=True))
    dataset[0]