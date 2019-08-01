import atexit
import os
import os.path as osp

import numpy as np
import skimage.io
import shutil

import instanceseg.utils.script_setup as script_utils
from instanceseg.utils import parse
from instanceseg.analysis import visualization_utils
from instanceseg.utils.script_setup import setup_train, configure
import debugging.helpers as debug_helper
from instanceseg.train import trainer

here = osp.dirname(osp.abspath(__file__))


def query_remove_logdir(logdir):
    from instanceseg.utils import misc
    if misc.y_or_n_input('Remove {}?'.format(logdir), default='n') == 'y':
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
    trainer = setup_train(args.dataset, cfg, out_dir, sampler_cfg, gpu=args.gpu, checkpoint_path=args.resume,
                          semantic_init=args.semantic_init)

    if cfg['debug_dataloader_only']:
        debug_helper.debug_dataloader(trainer, split='train')
        atexit.unregister(query_remove_logdir)
        return

    print('Evaluating final model')
    metrics = run(trainer)
    print('''\
        Accuracy: {0}
        Accuracy Class: {1}
        Mean IU: {2}
        FWAV Accuracy: {3}'''.format(*metrics))
    atexit.unregister(query_remove_logdir)


def run(trainer: trainer.Trainer):
    trainer.train()
    val_loss, eval_metrics, (segmentation_visualizations, score_visualizations) = \
        trainer.validate_split(should_export_visualizations=False)
    viz = visualization_utils.get_tile_image(segmentation_visualizations)
    skimage.io.imsave(os.path.join(here, 'viz_evaluate.png'), viz)
    eval_metrics = np.array(eval_metrics)
    eval_metrics *= 100
    return eval_metrics


if __name__ == '__main__':
    main()
