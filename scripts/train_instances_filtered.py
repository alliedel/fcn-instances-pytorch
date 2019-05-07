import os
import os.path as osp

import numpy as np
import skimage.io

import instanceseg.utils.scripts as script_utils
from instanceseg.analysis import visualization_utils
from instanceseg.utils.scripts import setup, configure

here = osp.dirname(osp.abspath(__file__))


def parse_args():
    args, cfg_override_args = script_utils.parse_args()
    return args, cfg_override_args


def main():
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args()
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args)
    trainer = setup(args.dataset, cfg, out_dir, sampler_cfg, gpu=args.gpu, resume=args.resume,
                    semantic_init=args.semantic_init)

    print('Evaluating final model')
    metrics = run(trainer)
    print('''\
        Accuracy: {0}
        Accuracy Class: {1}
        Mean IU: {2}
        FWAV Accuracy: {3}'''.format(*metrics))


def run(trainer):
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
