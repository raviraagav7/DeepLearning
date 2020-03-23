from __future__ import print_function
import argparse
import os
from azureml.core import Workspace
from azureml.core import Experiment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from create_azureml_compute import create_compute_target
from azureml.train.dnn import PyTorch
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import GridParameterSampling, choice, uniform, RandomParameterSampling
from azureml.train.hyperdrive import BanditPolicy


def upload_data2azure(ds, src_dir, target_dir='data_sow_class_v2'):
    """
        This function uploads the data from the source directory(src_dir) to
        Azure Data storage under the target directory.
        Args:
              ds: (object) Data store object.
              src_dir: (str) The directory where the Training and Validation Image Folders are present.
              target_dir:(str) The name of the Directory on Azure Storage.
        Return:
            None
    """
    ds.upload(src_dir=src_dir, target_path=target_dir, overwrite=False, show_progress=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Classifier using Torch')

    parser.add_argument('--src_dir',
                        help='Directory where the Training and Validation Image Folders are present.',
                        required=False,
                        default='/Users/srinivasraviraagav/Kespry-Code/convertLabel2Pipeline/SOW2_Data_v2')

    parser.add_argument('-g', '--grid',
                        help='Grid Parameter Sampling for hyper parameter range',
                        action='store_true')

    parser.add_argument('--compute_name',
                        help='Name of the compute target',
                        required=False,
                        default='ml-compute-high')

    parser.add_argument(
        '--output_dir',
        default='./outputs',
        type=str,
        help='The directory where the model weights will be stored')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1.e-3,
        help='Learning rate.'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.99,
        help='Momentum Rate.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs.'
    )

    parser.add_argument(
        '--is_pretrained',
        required=False,
        type=bool,
        default=True,
        help='To fine tune the network.'
    )

    parser.add_argument('--vm_size',
                        help='Type of instance. Defaults to STANDARD_NC6S_V2',
                        required=False, default="STANDARD_NC6S_V2")
    parser.add_argument('--vm_priority',
                        help='Priority of instance either dedicated or lowpriority. Defaults to lowpriority',
                        required=False, default='normalpriority')
    parser.add_argument('--max_nodes',
                        help='Max number of nodes in the cluster. Defaults to 4',
                        required=False, default=4, type=int)

    args = parser.parse_args()

    ws = Workspace.from_config(path=os.path.join(os.path.dirname(__file__), 'workspace_azure_ml_config.json'))
    print("Workspace: \n\tname: {}\n\tresource_group: {}\n\tlocation: {}\n\tsubscription_id: {}".format(
        ws.name, ws.resource_group, ws.location, ws.subscription_id))

    compute_target = create_compute_target(ws, args.compute_name, args.vm_size, args.vm_priority, args.max_nodes)

    ds = ws.get_default_datastore()

    upload_data2azure(ds, args.src_dir)

    experiment = Experiment(workspace=ws, name='train_torch_classifier')

    script_params = {
        '--data_dir': ds.path('data_sow_class_v2').as_mount(),
        '--output_dir': args.output_dir,
        '--batch_size': 32,
        '--model_architecture': 'mobile_net',
        '--learning_rate': args.learning_rate,
        '--momentum': args.momentum,
        '--epochs': args.epochs,
        '--is_pretrained': args.is_pretrained,
    }

    script_folder = os.path.join(os.path.dirname(__file__), 'torch', 'classification')

    estimator = PyTorch(source_directory=script_folder,
                        entry_script='train.py',
                        script_params=script_params,
                        compute_target=compute_target,
                        use_gpu=True,
                        pip_requirements_file='requirements.txt')

    # Hyper parameter tuning

    hyper_parameter_ranges = {
        '--learning_rate': uniform(1e-4, 2e-3),
        '--model_architecture': choice('wide_res_net_50', 'mobile_net', 'vgg_19', 'res_next', 'res_net101'),
        '--batch_size': choice(4, 8, 16)
    }

    if args.grid:
        param_sampling = GridParameterSampling(hyper_parameter_ranges)
    else:
        param_sampling = RandomParameterSampling(hyper_parameter_ranges)

    early_termination_policy = BanditPolicy(slack_factor=0.1,
                                            evaluation_interval=2,
                                            delay_evaluation=5)

    hyper_drive_run_config = HyperDriveConfig(estimator=estimator,
                                              hyperparameter_sampling=param_sampling,
                                              policy=early_termination_policy,
                                              primary_metric_name='val_acc',
                                              primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                              max_total_runs=30,
                                              max_concurrent_runs=4)

    run = experiment.submit(hyper_drive_run_config)
    run.wait_for_completion(show_output=True)
    # run = experiment.submit(estimator)
    # run.wait_for_completion(show_output=True)
