import atexit
import os
import os.path as osp
import sys
import time

import numpy as np

import debugging.dataloader_debug_utils as debug_helper
import instanceseg.utils.script_setup as script_utils
from instanceseg.analysis import visualization_utils
from instanceseg.train import trainer, validator
from instanceseg.utils import parse, misc
from instanceseg.utils.imgutils import write_np_array_as_img
from instanceseg.utils.misc import y_or_n_input
from instanceseg.utils.script_setup import setup_train, configure

here = osp.dirname(osp.abspath(__file__))

# DEBUG_WATCHER = True
DEBUG_WATCHER = False


def query_remove_logdir(logdir):
    from instanceseg.utils import misc
    import os
    import shutil
    if misc.y_or_n_input('Remove {}?'.format(logdir), default='n') == 'y':
        if os.path.exists(logdir):
            shutil.rmtree(logdir)


def parse_args(replacement_dict_for_sys_args=None):
    args, cfg_override_args = parse.parse_args_train(replacement_dict_for_sys_args)
    return args, cfg_override_args


def main(replacement_dict_for_sys_args=None):
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args(replacement_dict_for_sys_args)
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args)
    atexit.register(query_remove_logdir, out_dir)

    trainer_gpu = args.gpu
    watchingval_gpu = None if cfg['validation_gpu'] is None or len(cfg['validation_gpu']) == 0 \
        else int(cfg['validation_gpu'])

    if cfg['train_batch_size'] == 1 and len(trainer_gpu) > 1:
        print(misc.color_text('Batch size is 1; another GPU won\'t speed things up.  We recommend assigning the other '
                              'gpu to validation for speed: --validation_gpu <gpu_num>', 'WARNING'))

    trainer = setup_train(args.dataset, cfg, out_dir, sampler_cfg, gpu=trainer_gpu,
                          checkpoint_path=args.resume, semantic_init=args.semantic_init)

    if cfg['debug_dataloader_only']:
        n_debug_images = None if cfg['n_debug_images'] is None else int(cfg['n_debug_images'])
        debug_helper.debug_dataloader(trainer, split='train', n_debug_images=n_debug_images)
        atexit.unregister(query_remove_logdir)
    else:
        metrics = run(trainer, watchingval_gpu)
        # atexit.unregister(query_remove_logdir)
        if metrics is not None:
            print('''\
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}
                FWAV Accuracy: {3}'''.format(*metrics))
    return out_dir


def terminate_watcher(pid, writer):
    print('Terminating watcher')
    pid.terminate()
    writer.close()


def find_and_kill_watcher(my_trainer_logdir):
    print('Killing watcher')
    import subprocess
    pid = subprocess.Popen(['ps', 'ux'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = pid.communicate()
    processes = output.decode('ascii').split('\n')
    watch_processes = [proc for proc in processes if 'watch_and_validate' in proc]
    my_watch_processes = [proc for proc in watch_processes if os.path.basename(my_trainer_logdir) in proc]
    if len(my_watch_processes) > 1:
        raise Exception('Multiple watch processes to kill. That shouldnt happen')
    elif len(my_watch_processes) == 1:
        process_id = [s for s in my_watch_processes[0].split(' ') if s != ''][1]
        print('Killing watcher (process {})'.format(int(process_id)))
        subprocess.check_call(['kill', '{}'.format(int(process_id))])
    else:
        print('No validation watcher to kill.')


def run(my_trainer: trainer.Trainer, watching_validator_gpu=None, write_val_to_stdout=False):
    if watching_validator_gpu is not None:
        atexit.register(find_and_kill_watcher, my_trainer.exporter.out_dir)
        pid, pidout_filename, writer = validator.offload_validation_to_watcher(my_trainer, watching_validator_gpu,
                                                                               as_subprocess=not DEBUG_WATCHER,
                                                                               write_val_to_stdout=write_val_to_stdout)
        atexit.unregister(find_and_kill_watcher)
        atexit.register(terminate_watcher, pid, writer)
    else:
        pid = None
        pidout_filename = None
        writer = None

    try:
        my_trainer.train()
        misc.color_text('Training is complete!', 'OKGREEN')
        atexit.unregister(query_remove_logdir)
    except KeyboardInterrupt:
        if y_or_n_input('I\'ve stopped training.  Finish script?', default='y') == 'n':
            raise

    if pid is not None:
        with open(pidout_filename, 'rb', 1) as reader:
            while pid.poll() is None:
                if my_trainer.t_val.finished():
                    misc.color_text('Validation is complete!', 'OKGREEN')
                if write_val_to_stdout:
                    import ipdb;
                    ipdb.set_trace()
                    sys.stdout.write(reader.read())
                print('Models: {}'.format(my_trainer.t_val.get_trained_model_list()))
                print('Finished: {}'.format(my_trainer.t_val.get_finished_files()))
                print('All: {}'.format(my_trainer.t_val.get_watcher_log_files()))
                time.sleep(0.5)
            # Read the remaining
            if write_val_to_stdout:
                sys.stdout.write(reader.read())

    print('Evaluating final model')
    val_loss, eval_metrics, (segmentation_visualizations, score_visualizations) = \
        my_trainer.validate_split(should_export_visualizations=False)
    if eval_metrics is not None:
        eval_metrics = np.array(eval_metrics)
        eval_metrics *= 100
    viz = visualization_utils.get_tile_image(segmentation_visualizations)
    write_np_array_as_img(os.path.join(here, 'viz_evaluate.png'), viz)

    return eval_metrics


if __name__ == '__main__':
    main()
