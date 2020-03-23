import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from train.torch.segmentation.dataloader import Dataset
from train.torch.segmentation.utils import get_training_augmentation, get_validation_augmentation, get_preprocessing
import argparse


class SegmentationNet:

    def __init__(self, base_data_dir, classes=['wireframe'], epoch=40,
                 batch_size=4, learning_rate=0.0001, output_dir='./outputs'):

        self.base_data_dir = base_data_dir
        self.x_train_dir = os.path.join(base_data_dir, 'train', 'ColorImages')
        self.y_train_dir = os.path.join(base_data_dir, 'train', 'SegmentationMask')

        self.x_valid_dir = os.path.join(base_data_dir, 'val', 'ColorImages')
        self.y_valid_dir = os.path.join(base_data_dir, 'val', 'SegmentationMask')
        self.output_dir = output_dir

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ENCODER = 'se_resnext50_32x4d'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = classes
        self.ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.preprocessing_fn = None
        self.metrics = None
        self.optimizer = None

        self.build_model()

    def build_model(self):
        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name=self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=len(self.CLASSES),
            activation=self.ACTIVATION,
        )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)

        # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

        loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        self.optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=self.learning_rate),
        ])

    def run(self):
        train_dataset = Dataset(
            self.x_train_dir,
            self.y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.CLASSES,
        )

        valid_dataset = Dataset(
            self.x_valid_dir,
            self.y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)

        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.DEVICE,
            verbose=True,
        )

        # train model for 40 epochs

        max_score = 0

        for i in range(0, self.epoch):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, '{}/best_model_run2.pth'.format(self.output_dir))
                print('Model saved!')

            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a segmentation Model')

    parser.add_argument('--data_dir', type=str,
                        default='./WireframeData',
                        help='Path to the data directory')

    parser.add_argument('--learning_rate', required=False,
                        type=float, default=0.001, help='Learning Rate.')

    parser.add_argument('--epochs', required=False,
                        type=int, default=35, help='Number of epochs.')

    parser.add_argument('--batch_size', required=False, type=int,
                        default=4, help='Batch Size.')

    parser.add_argument('--classes', required=False, type=list,
                        default=['wireframe'], help='List of Classes.')

    parser.add_argument('--output_dir', required=False,
                        type=str, default='./outputs',
                        help='The directory where the weights needs to be saved')

    args = parser.parse_args()

    o_seg = SegmentationNet(base_data_dir=args.data_dir,
                            classes=args.classes,
                            epoch=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            output_dir=args.output_dir)
    o_seg.run()

