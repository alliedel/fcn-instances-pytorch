import argparse
import os
import yaml

from instanceseg.train.filewatcher import WatchingValidator
from instanceseg.utils import script_setup
from scripts.configurations import sampler_cfg_registry


def main(trainer_logdir, gpu):
    trainer_config_file = os.path.join(trainer_logdir, 'config.yaml')
    train_cfg = yaml.safe_load(open(trainer_config_file, 'r'))
    train_outdir = os.path.dirname(trainer_config_file)
    watch_val_outdir = os.path.join(os.path.dirname(trainer_config_file), 'WatchingValidator')
    validator_kwargs = {
        'dataset_type': train_cfg['dataset'],
        'train_cfg': train_cfg,
        'sampler_cfg': sampler_cfg_registry.get_sampler_cfg_set(train_cfg['sampler']),
        'out_dir': watch_val_outdir,
        'init_model_checkpoint_path': None
    }

    validator = script_setup.setup_validator(**validator_kwargs, gpu=gpu)
    watching_validator = WatchingValidator(validator, os.path.join(train_outdir, 'model_checkpoints'))
    watching_validator.start()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, required=True)
    parser.add_argument('trainer_logdir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    _args = parse_args()
    main(_args.trainer_logdir, _args.gpu)
