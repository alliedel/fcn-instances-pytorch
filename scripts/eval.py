import os
import os.path as osp
import torch
import yaml

import debugging.helpers as debug_helper
import instanceseg.utils.script_setup as script_utils
import utils.parse
from instanceseg.factory.models import get_problem_config
import instanceseg.factory.trainers as trainer_factory
from instanceseg.utils.instance_utils import InstanceProblemConfig
from instanceseg.utils.script_setup import configure
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS
from instanceseg.utils.script_setup import setup_common
import instanceseg.factory.samplers as sampler_factory
from instanceseg.datasets import dataset_generator_registry
import torch.utils.data


here = osp.dirname(osp.abspath(__file__))



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


def setup(logdir, args, cfg, sampler_cfg, out_dir):
    checkpoint, cuda, dataloaders, model, problem_config, start_epoch, start_iteration = \
        setup_common(dataset_type=args.dataset, cfg=cfg, gpu=args.gpu, resume=args.resume, sampler_cfg=sampler_cfg,
                     semantic_init=args.semantic_init)
    # Load model
    model_checkpoint_pth = os.path.join(logdir, 'checkpoint.pth.tar')
    test_model = torch.load(model_checkpoint_pth)

    # Load configuration options
    train_config_path = os.path.join(logdir, 'config.yaml')
    test_config_options = get_config_options_from_train_config(train_config_path)

    trainer = trainer_factory.get_trainer(cfg, cuda, model, dataloaders, problem_config, out_dir, optim=None,
                                          scheduler=None)
    # Dataloaders
    test_dataloader = get_dataloader()

    # Instance problem configuration
    with os.path.join(logdir, 'instance_problem_config.yaml') as instance_config_file:
        if os.path.exists(instance_config_file):
            problem_config = InstanceProblemConfig.load(instance_config_file)
        else:
            semantic_class_names =
            problem_config = get_problem_config(semantic_class_names, n_instances_per_class,
                                                map_to_semantic=cfg['map_to_semantic'])
    return trainer


def parse_args(replacement_dict_for_sys_args=None):
    args, cfg_override_args = utils.parse.parse_args_train(replacement_dict_for_sys_args)
    return args, cfg_override_args


def main(replacement_dict_for_sys_args=None):
    logdir = '/usr0/home/adelgior/code/experimental-instanceseg/scripts/logs/synthetic/' \
             'train_instances_filtered_2019-06-24-163353_VCS-8df0680'
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args(replacement_dict_for_sys_args)
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args)

    trainer = setup(logdir, args, cfg, sampler_cfg, out_dir)

    if cfg['debug_dataloader_only']:
        debug_helper.debug_dataloader(trainer)
        return


if __name__ == '__main__':
    main()
