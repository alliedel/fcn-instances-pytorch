import atexit
import io
import os
import os.path as osp
import subprocess
import time
import sys

import numpy as np

import debugging.dataloader_debug_utils as debug_helper
import instanceseg.utils.script_setup as script_utils
from instanceseg.analysis import visualization_utils
from instanceseg.train import trainer
from instanceseg.utils import parse
from instanceseg.utils.imgutils import write_np_array_as_img
from instanceseg.utils.misc import y_or_n_input
from instanceseg.utils.script_setup import setup_train, configure

here = osp.dirname(osp.abspath(__file__))


DEBUG_WATCHER = False

if DEBUG_WATCHER:
    from scripts import watch_and_validate
else:
    watch_and_validate = None


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
    if len(args.gpu) == 1:
        trainer_gpu = args.gpu
        watchingval_gpu = None
    else:
        trainer_gpu = [g for g in args.gpu[:-1]]
        watchingval_gpu = args.gpu[-1]
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args)
    atexit.register(query_remove_logdir, out_dir)

    trainer = setup_train(args.dataset, cfg, out_dir, sampler_cfg, gpu=trainer_gpu,
                          checkpoint_path=args.resume, semantic_init=args.semantic_init)

    if cfg['debug_dataloader_only']:
        n_debug_images = None if cfg['n_debug_images'] is None else int(cfg['n_debug_images'])
        debug_helper.debug_dataloader(trainer, split='train', n_debug_images=n_debug_images)
        atexit.unregister(query_remove_logdir)
    else:
        print('Evaluating final model')
        metrics = run(trainer, watchingval_gpu)
        # atexit.unregister(query_remove_logdir)
        if metrics is not None:
            print('''\
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}
                FWAV Accuracy: {3}'''.format(*metrics))
    return out_dir


def start_watcher(my_trainer, watching_validator_gpu, as_subprocess=(not DEBUG_WATCHER)):
    if watching_validator_gpu is not None:
        pidout_filename = os.path.join(my_trainer.exporter.out_dir, 'watcher_output.log')
        writer = io.open(pidout_filename, 'wb')
        if not as_subprocess:
            watch_and_validate.main(my_trainer.exporter.out_dir, watching_validator_gpu)
            return
        else:
            pid = subprocess.Popen(['python', 'scripts/watch_and_validate.py', my_trainer.exporter.out_dir, '--gpu',
                                    '{}'.format(watching_validator_gpu)], stdout=writer)
    else:
        pid = None
        pidout_filename = None
        writer = None
    return pid, pidout_filename, writer


def terminate_watcher(pid, writer):
    pid.terminate()
    writer.close()


def run(my_trainer: trainer.Trainer, watching_validator_gpu=None):
    pid, pidout_filename, writer = start_watcher(my_trainer, watching_validator_gpu)
    atexit.register(terminate_watcher, pid, writer)

    try:
        my_trainer.train()
        atexit.unregister(query_remove_logdir)
    except KeyboardInterrupt:
        if y_or_n_input('I\'ve stopped training.  Finish script?', default='y') == 'n':
            raise

    if pid is not None:
        with io.open(pidout_filename, 'rb', 1) as reader:
            while pid.poll() is None:
                sys.stdout.write(reader.read())
                time.sleep(0.5)
            # Read the remaining
            sys.stdout.write(reader.read())

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
