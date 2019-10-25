import argparse
import os

import instanceseg.train.validator
from instanceseg.train import filewatcher
from instanceseg.train import trainer


def main(trainer_logdir, gpu, starting_model_checkpoint=None):
    validator = instanceseg.train.validator.get_validator(trainer_logdir, gpu, starting_model_checkpoint)
    watching_validator = filewatcher.WatchingValidator(validator, os.path.join(trainer_logdir, 'model_checkpoints'))
    watching_validator.start()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, required=True)
    parser.add_argument('--starting_model_checkpoint', type=str, default=None)
    parser.add_argument('trainer_logdir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    _args = parse_args()
    main(_args.trainer_logdir, _args.gpu)
