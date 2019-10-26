import os

import yaml

from instanceseg.utils import script_setup, misc
from scripts import watch_and_validate
from scripts.configurations import sampler_cfg_registry
import subprocess
import datetime

WATCH_VAL_SUBDIR = 'watching_validator'


def start_screen_session_with_cmd(cmd, session_name=None, new_window_name='new_window_name'):
    # screen -dm -S screen_name
    # -dm -- in detached state
    # screen -S automate_me -X stuff 'echo HELLO WORLD'$(echo -ne '\015')
    #
    now_str = datetime.datetime.now().strftime('%m_%d_%H-%M-%S')
    session_name = session_name or 'autoscreen_{}'.format(now_str)
    new_window_name = new_window_name or 'autowindow_{}'.format(now_str)
    window_creator = ['screen', '-S', session_name, '-t', new_window_name, '-A', '-d', '-m']
    print('Ran:\n' + ' '.join(window_creator))
    subprocess.check_output(window_creator)
    # Reload direnv
    cmd_opener = ['screen', '-S', session_name, '-p', new_window_name, '-X', 'stuff', 'direnv reload\n']
    subprocess.check_output(cmd_opener, shell=False)  # If this fails, it's likely a name overload.

    cmd_opener = ['screen', '-S', session_name, '-p', new_window_name, '-X', 'stuff', cmd + '\n']
    print('Ran:\n' + ' '.join(cmd_opener))
    pid2 = subprocess.check_output(cmd_opener, shell=False)
    return pid2


def offload_validation_to_watcher(my_trainer, watching_validator_gpu, as_subprocess=True,
                                  write_val_to_stdout=False):
    assert my_trainer.t_val is None, 'Watcher already exists'
    starting_model_checkpoint = my_trainer.exporter.save_checkpoint(my_trainer.state.epoch,
                                                                    my_trainer.state.iteration, my_trainer.model,
                                                                    my_trainer.optim, my_trainer.best_mean_iu,
                                                                    mean_iu=None)
    pidout_filename = os.path.join(my_trainer.exporter.out_dir, 'watcher_output_subprocess.log')
    writer = open(pidout_filename, 'wb')
    if not as_subprocess:  # debug
        validator = get_validator(my_trainer.exporter.out_dir, watching_validator_gpu,
                                  starting_model_checkpoint)
        watch_and_validate.main(my_trainer.exporter.out_dir, watching_validator_gpu,  # loops forever
                                starting_model_checkpoint=starting_model_checkpoint)
        return
    else:
        # pid = subprocess.Popen(['screen', '-d', '-RR' '-d', '-m', 'python', 'scripts/watch_and_validate.py',
        #                         my_trainer.exporter.out_dir,
        #                         '--gpu', '{}'.format(watching_validator_gpu), '--starting_model_checkpoint',
        #                         starting_model_checkpoint], stdout=writer, stderr=subprocess.STDOUT)
        cmd = ' '.join(['python', 'scripts/watch_and_validate.py',
                        my_trainer.exporter.out_dir, '--gpu', '{}'.format(watching_validator_gpu),
                        '--starting_model_checkpoint', starting_model_checkpoint])
        pid = start_screen_session_with_cmd(cmd)
    misc.color_text('Offloaded validation to GPU {}'.format(watching_validator_gpu), color='OKBLUE')
    my_trainer.skip_validation = True
    my_trainer.t_val = my_trainer.get_validation_progress_bar()
    try:
        assert my_trainer.t_val is not None
    except AssertionError:
        print('Watcher didnt start.  Error log at {}'.format(pidout_filename))
        raise
    return pid, pidout_filename, writer


def get_validator(trainer_logdir, gpu, starting_model_checkpoint=None):
    trainer_config_file = os.path.join(trainer_logdir, 'config.yaml')
    watch_val_outdir = os.path.join(os.path.dirname(trainer_config_file), 'watching_validator')
    train_cfg = yaml.safe_load(open(trainer_config_file, 'r'))
    if not os.path.exists(watch_val_outdir):
        os.mkdir(watch_val_outdir)

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
