import datetime
import os
import time
import sys
import torch
import torch.utils.data
import argparse
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import ssl
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import random
import matplotlib.patches as patches

sys.path.append('.')
from utils.engine import train_one_epoch, evaluate
import utils.transforms as T
import utils.utils as utils
# from data import PedData
from data_loader import SOW2Dataset

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context


class Trainer:

    def __init__(self, data_dir, batch_size, model_architecture, num_classes,
                 learning_rate, epochs, momentum=0.9,
                 weight_decay=1e-4, lr_step_size=8,
                 lr_steps=[16, 22], lr_gamma=0.1,
                 print_freq=20, output_dir='.',
                 start_epoch=0, is_pretrained=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.model_architecture = model_architecture
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma
        self.print_freq = print_freq
        self.output_dir = output_dir
        self.start_epoch = start_epoch
        self.is_pretrained = is_pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(self.is_pretrained)
        self.num_classes = num_classes + 1 # num_classes + Background
        if self.output_dir:
            utils.mkdir(self.output_dir)

    def __get_transform(self, train):
        transforms = list()
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def __get_model(self):
        if self.model_architecture == 'resnet50':
            # load a model pre-trained pre-trained on COCO
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.is_pretrained)
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        elif self.model_architecture == 'mobilenet':
            # load a pre-trained model for classification and return only the features
            backbone = torchvision.models.mobilenet_v2(pretrained=self.is_pretrained).features
            # FasterRCNN needs to know the number of
            # output channels in a backbone. For mobilenet_v2, it's 1280
            # so we need to add it here
            backbone.out_channels = 1280

            # let's make the RPN generate 5 x 3 anchors per spatial
            # location, with 5 different sizes and 3 different aspect
            # ratios. We have a Tuple[Tuple[int]] because each feature
            # map could potentially have different sizes and
            # aspect ratios
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),))

            # let's define what are the feature maps that we will
            # use to perform the region of interest cropping, as well as
            # the size of the crop after rescaling.
            # if your backbone returns a Tensor, featmap_names is expected to
            # be [0]. More generally, the backbone should return an
            # OrderedDict[Tensor], and in featmap_names you can choose which
            # feature maps to use.
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                            output_size=7,
                                                            sampling_ratio=2)

            # put the pieces together inside a FasterRCNN model
            self.model = FasterRCNN(backbone,
                                    num_classes=self.num_classes,
                                    rpn_anchor_generator=anchor_generator,
                                    box_roi_pool=roi_pooler)

    @torch.no_grad()
    def predict(self, image_path, classes, conf_thresh=0.5):
        weights_file_path = os.listdir(self.output_dir)
        self.__get_model()
        self.model.to(self.device)
        checkpoint = torch.load(os.path.join(self.output_dir, weights_file_path[-1]), map_location='cpu')
        self.model.load_state_dict(checkpoint)

        self.model.eval()

        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        transformed_img = transform(img)
        transformed_img = transformed_img.to(self.device)
        # transformed_img = transformed_img.to(self.device)
        outputs = self.model([transformed_img])
        cpu_device = torch.device("cpu")
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        print(outputs)

        # Get bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        img = np.array(img)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        if outputs is not None:
            unique_labels = outputs[0]['labels'].unique()
            n_cls_pred = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_pred)
            for bbox, cls_pred, conf in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
                if conf > conf_thresh:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    box = patches.Rectangle((bbox[0], bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]),
                                            linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(box)
                    plt.text(bbox[0], bbox[1], s='{} : {:.2f}'.format(classes[cls_pred], conf),
                             color='white', verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
        plt.axis('off')
        plt.show()
        self.model.train()

    def main(self):
        # Data loading code
        print("Loading data")

        # use our dataset and defined transformations
        # dataset = PedData(self.data_dir, self.__get_transform(train=True))
        dataset = SOW2Dataset('Underwater Project', os.path.join(self.data_dir, 'train'),
                              self.__get_transform(train=True), mode='train')
        # dataset_test = PedData(self.data_dir, self.__get_transform(train=False))
        dataset_test = SOW2Dataset('Underwater Project', os.path.join(self.data_dir, 'test'),
                                   self.__get_transform(train=False), mode='test')

        print(len(dataset))

        # # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        print("Creating data loaders")
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, self.batch_size, drop_last=True)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=4,
            collate_fn=utils.collate_fn)

        print("Creating model")
        self.__get_model()
        self.model.to(self.device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(
            params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        print("Start training")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, args.print_freq)

            lr_scheduler.step()
            if self.output_dir and epoch == self.epochs-1:
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, 'model_{}_{}.pth'.format(epoch, self.model_architecture)))
            # evaluate after every epoch
            evaluate(self.model, data_loader_test, device=self.device)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a detector network.')

    parser.add_argument('--data_path', default='/home/ravi/dataset/batch-1', help='dataset')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.002, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_step_size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output_dir', default='./models-sow2', help='path where to save')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes.')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    o_train = Trainer(data_dir=args.data_path, model_architecture=args.model,
                      batch_size=args.batch_size, num_classes=args.num_classes,
                      epochs=args.epochs, learning_rate=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay, lr_step_size=args.lr_step_size,
                      lr_steps=args.lr_steps, lr_gamma=args.lr_gamma, print_freq=args.print_freq,
                      output_dir=args.output_dir, start_epoch=args.start_epoch,
                      is_pretrained=True)

    # o_train.main()

    o_train.predict(image_path='/home/ravi/dataset/batch-1/test/N0302_1_Starboard_image_D2018-09-13T22-21-13-911510Z_2_Starboard_0007_Anode.jpg',
                    classes=['Background', 'Anode', 'Boulder', 'Debris'])
