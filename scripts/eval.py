import os
import torch
import yaml

from instanceseg.factory.models import get_problem_config
from instanceseg.utils.instance_utils import InstanceProblemConfig
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS


def keys_to_transfer_from_train_to_test():
    keys_to_transfer = []
    for k in PARAM_CLASSIFICATIONS.data:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.debug:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.problem_config:
        keys_to_transfer.append(k)

    keys_to_transfer.remove('train_batch_size')

    return keys_to_transfer


def get_config_options_from_train_config(train_config_path):
    train_config = yaml.safe_load(open(train_config_path, 'r'))
    test_config_options = {
        k: v for k, v in train_config.items() if k in keys_to_transfer_from_train_to_test()
    }
    test_config_options['test_batch_size'] = test_config_options.pop('val_batch_size')
    return test_config_options


def setup(logdir):
    # Load model
    model_checkpoint_pth = os.path.join(logdir, 'checkpoint.pth.tar')
    test_model = torch.load(model_checkpoint_pth)

    # Load configuration options
    train_config_path = os.path.join(logdir, 'config.yaml')
    test_config_options = get_config_options_from_train_config(train_config_path)

    # Dataloaders


    # Instance problem configuration
    with os.path.join(logdir, 'instance_problem_config.yaml') as instance_config_file:
        if os.path.exists(instance_config_file):
            problem_config = InstanceProblemConfig.load(instance_config_file)
        else:
            semantic_class_names =
            problem_config = get_problem_config(semantic_class_names, n_instances_per_class,
                                                map_to_semantic=cfg['map_to_semantic'])


def main():
    logdir = '/usr0/home/adelgior/code/experimental-instanceseg/scripts/logs/synthetic/' \
             'train_instances_filtered_2019-06-24-163353_VCS-8df0680'

    evaluator = setup(logdir)

    print('Done!')


if __name__ == '__main__':
    main()
