from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
from tqdm import tqdm
from enum import Enum
from azureml.core import Run
plt.ion()
run = Run.get_context()


class ModelArchitecture(Enum):
    WIDE_RES_NET_50 = 'wide_res_net_50'
    MOBILE_NET_V2 = 'mobile_net'
    VGG_19 = 'vgg_19'
    RES_NEXT = 'res_next'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Trainer:

    def __init__(self, data_dir, batch_size, model_architecture, learning_rate,
                 epochs, momentum, is_pretrained, output_dir, experiment='Ant_and_Bees'):
        """

        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.model_architecture = model_architecture
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.is_pretrained = is_pretrained
        self.output_dir = output_dir
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.exp_lr_scheduler = None
        self.writer = SummaryWriter(log_dir='runs/{}'.format(experiment))
        self.data_transforms = {'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]), }

        self.image_data = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                           for x in ['train', 'val']}

        self.data_loader = {x: torch.utils.data.DataLoader(self.image_data[x], batch_size=self.batch_size,
                                                           shuffle=True, num_workers=4) for x in ['train', 'val']}

        self.data_sizes = {x: len(self.image_data[x]) for x in ['train', 'val']}

        self.class_names = self.image_data['train'].classes

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__build_model()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print('Created Folder.')

    @staticmethod
    def im_show(inp, title=None):
        """
            This function shows a sample image from a tensor.

            Args:
                inp: (tensor) Image Tensor
                title: (str) Title to be displayed over the image.
        """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(10)

    def __build_model(self):
        if self.model_architecture == ModelArchitecture.WIDE_RES_NET_50.value:
            self.model = models.wide_resnet50_2(pretrained=self.is_pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_names))

        elif self.model_architecture == ModelArchitecture.RES_NEXT.value:
            self.model = models.resnext101_32x8d(pretrained=self.is_pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_names))

        elif self.model_architecture == ModelArchitecture.VGG_19.value:
            self.model = models.vgg19_bn(pretrained=self.is_pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, len(self.class_names))

        elif self.model_architecture == ModelArchitecture.MOBILE_NET_V2.value:
            self.model = models.mobilenet_v2(pretrained=self.is_pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, len(self.class_names))

        self.model = self.model.to(self.device)

        # self.criterion = nn.BCEWithLogitsLoss() if len(self.class_names) == 2 else nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train_model(self):
        since = time.time()

        best_acc = 0.0
        run.log('Batch Size', int(self.batch_size))
        run.log('Model Architecture', str(self.model_architecture))
        run.log('Learning Rate', float(self.learning_rate))
        run.log('Momentum', float(self.momentum))

        for epoch in range(self.epochs):

            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase is 'train':
                    self.model.train()  # Set model to training mode.
                else:
                    self.model.eval()  # Set model to evaluate mode.

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(self.data_loader[phase], ncols=50):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    # track history if only in train

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase is 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item()# * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.exp_lr_scheduler.step()

                epoch_loss = running_loss / self.data_sizes[phase]
                epoch_acc = running_corrects.double() / self.data_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                run.log('{}_loss'.format(phase), float(epoch_loss))
                run.log('{}_acc'.format(phase), float(epoch_acc))

                if phase == 'train':
                    self.writer.add_scalar('training_loss', epoch_loss, epoch)
                    self.writer.add_scalar('training_acc', epoch_acc, epoch)
                else:
                    self.writer.add_scalar('validation_loss', epoch_loss, epoch)
                    self.writer.add_scalar('validation_acc', epoch_acc, epoch)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir,
                                                                     'best_weights_{}_batch_{}_lr_{}.pth'.format(
                                                                         self.model_architecture,
                                                                         self.batch_size, self.learning_rate)))
                    print(os.listdir(self.output_dir))

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # Model loaded with best weights.
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir,
                                                           'best_weights_{}_batch_{}_lr_{}.pth'.format(
                                                               self.model_architecture, self.batch_size,
                                                               self.learning_rate))))

    def visualize_model(self, num_images=6):

        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (input_data, label_data) in enumerate(self.data_loader['val']):
                input_data = input_data.to(self.device)
                _ = label_data.to(self.device)

                outputs = self.model(input_data)

                _, predictions = torch.max(outputs, 1)

                for j in range(input_data.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[predictions[j]]))
                    self.im_show(input_data.cpu().data[j])

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return
            self.model(mode=was_training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory where the Images are present.')

    parser.add_argument('--batch_size', required=False, type=int,
                        default=4, help='Batch Size.')

    parser.add_argument('--model_architecture', required=False, type=str,
                        default='res_next',
                        help='Backbone Model. (Example. "wide_res_net_50", "res_next", "mobile_net", "vgg19")')

    parser.add_argument('--learning_rate', required=False,
                        type=float, default=0.001, help='Learning Rate.')

    parser.add_argument('--epochs', required=False,
                        type=int, default=25, help='Number of epochs.')

    parser.add_argument('--momentum', required=False,
                        type=float, default=0.9, help='Momentum.')

    parser.add_argument('--is_pretrained', required=False,
                        type=bool, default=True, help='To fine tune the network.')

    parser.add_argument('--output_dir', required=False,
                        type=str, default='./model_weight', help='The directory where the weights needs to be saved')

    args = parser.parse_args()

    assert ModelArchitecture.has_value(args.model_architecture), 'Please provide the right architecture.'
    o_train = Trainer(args.data_dir, args.batch_size, args.model_architecture,
                      args.learning_rate, args.epochs, args.momentum, args.is_pretrained, args.output_dir)

    # To display the image.
    # inputs, classes = next(iter(o_train.data_loader['train']))
    # out = torchvision.utils.make_grid(inputs)
    # o_train.im_show(out, title=[o_train.class_names[x] for x in classes])

    o_train.train_model()
    # o_train.visualize_model()

