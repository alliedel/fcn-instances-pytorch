import os

import yaml

from instanceseg.utils import script_setup, misc
from scripts import watch_and_validate
from scripts.configurations import sampler_cfg_registry
import subprocess


WATCH_VAL_SUBDIR = 'watching_validator'


def offload_validation_to_watcher(self, watching_validator_gpu, as_subprocess=True,
                                  write_val_to_stdout=False):
    assert self.t_val is None, 'Watcher already exists'
    starting_model_checkpoint = self.exporter.save_checkpoint(self.state.epoch,
                                                              self.state.iteration, self.model,
                                                              self.optim, self.best_mean_iu,
                                                              mean_iu=None)
    pidout_filename = os.path.join(self.exporter.out_dir, 'watcher_output_subprocess.log')
    writer = open(pidout_filename, 'wb')
    if not as_subprocess:  # debug
        validator = get_validator(self.exporter.out_dir, watching_validator_gpu,
                                  starting_model_checkpoint)
        watch_and_validate.main(self.exporter.out_dir, watching_validator_gpu,  # loops forever
                                starting_model_checkpoint=starting_model_checkpoint)
        return
    else:
        pid = subprocess.Popen(['python', 'scripts/watch_and_validate.py',
                                os.path.join(self.exporter.out_dir, WATCH_VAL_SUBDIR),
                                '--gpu', '{}'.format(watching_validator_gpu), '--starting_model_checkpoint',
                                starting_model_checkpoint], stdout=writer, stderr=subprocess.STDOUT)
    misc.color_text('Offloaded validation to GPU {}'.format(watching_validator_gpu), color='OKBLUE')
    self.skip_validation = True
    self.t_val = self.get_validation_progress_bar()
    assert self.t_val is not None, ''
    return pid, pidout_filename, writer


def get_validator(trainer_logdir, gpu, starting_model_checkpoint=None):
    trainer_config_file = os.path.join(trainer_logdir, 'config.yaml')
    train_cfg = yaml.safe_load(open(trainer_config_file, 'r'))
    watch_val_outdir = os.path.join(os.path.dirname(trainer_config_file), 'watching_validator')
    if starting_model_checkpoint is not None:
        assert os.path.isfile(starting_model_checkpoint)
    else:
        starting_model_checkpoint = os.path.join(trainer_logdir, 'checkpoint.pth.tar')
    sampler_cfg = sampler_cfg_registry.get_sampler_cfg_set(train_cfg['sampler'])
    validator_kwargs = {
        'dataset_type': train_cfg['dataset'],
        'train_cfg': train_cfg,
        'sampler_cfg': sampler_cfg,
        'out_dir': watch_val_outdir,
        'init_model_checkpoint_path': starting_model_checkpoint
    }
    validator = script_setup.setup_validator(**validator_kwargs, gpu=gpu)
    assert 'train_for_val' in validator.dataloaders.keys()
    assert validator.state.iteration is not None
    assert validator.exporter.conservative_export_decider.base_interval is not None
    return validator