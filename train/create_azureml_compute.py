from __future__ import print_function
import os
import argparse
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


def create_compute_target(ws, compute_name, vm_size, vm_priority, max_nodes):
    """

        This function creates a compute target in AzureML Machine learning VM based on the "workspace_azure_ml_config.json" passed.

        Args:
            ws: (object) Workspace object loaded from the "workspace_azure_ml_config.json".
            compute_name: (str) Name of the AML compute that needs to be created within the workspace.
            vm_size: (str) Type of Virtual Machine Instance.
            vm_priority: (str) Priority of creating the Virtual Machine Instance.
            max_nodes: (int) Max number of nodes in the cluster.

        Returns:
            Compute target object that is created.
    """
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print('Found compute target: ' + compute_name)
    except ComputeTargetException:
        print('Creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                    vm_priority=vm_priority,
                                                                    min_nodes=0,
                                                                    max_nodes=max_nodes,
                                                                    idle_seconds_before_scaledown=60)
        # create the compute target
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        # Can poll for a minimum number of nodes and for a specific timeout.
        # If no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion( show_output=True, min_node_count=None, timeout_in_minutes=20)

        # For a more detailed view of current cluster status, use the 'status' property
        print(compute_target.status.serialize())

    return compute_target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an Azure AML Compute for model training.")

    parser.add_argument('--name',
                        help='Name of the AML compute',
                        required=True)
    parser.add_argument('--vm_size',
                        help='Type of instance. Defaults to STANDARD_NC6S_V2',
                        required=False, default="STANDARD_NC6S_V2")
    parser.add_argument('--vm_priority',
                        help='Priority of instance either dedicated or lowpriority. Defaults to lowpriority',
                        required=False, default='lowpriority')
    parser.add_argument('--max_nodes',
                        help='Max number of nodes in the cluster. Defaults to 4',
                        required=False, default=4, type=int)
    args = parser.parse_args()

    ws = Workspace.from_config(path=os.path.join(os.path.dirname(__file__), '../workspace_azure_ml_config.json'))

    print("Workspace: \n\tname: {}\n\tresource_group: {}\n\tlocation: {}\n\tsubscription_id: {}".format(
        ws.name, ws.resource_group, ws.location, ws.subscription_id))

    aml_compute = create_compute_target(ws, args.name, args.vm_size, args.vm_priority, args.max_nodes)

    print(aml_compute)
