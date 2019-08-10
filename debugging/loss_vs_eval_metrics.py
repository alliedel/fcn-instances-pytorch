import os

from instanceseg.utils import parse
from scripts import test
from instanceseg.utils import script_setup

train_logdir = 'scripts/logs/synthetic/train_2019-08-04-212241_VCS-160c04f_BACKBONE-fcn8'
stored_models_dir = os.path.join(train_logdir, 'model_checkpoints')
# tf_events_file = glob.glob(os.path.join(train_logdir, 'events.out.tfevents.*'))[0]


all_saved_models = os.path.join(stored_models_dir, '*.pth.tar')
n_models_to_evaluate = 10

logdir = train_logdir.rstrip('/')
dataset_name = os.path.basename(os.path.dirname(logdir))

split = 'val'
sampler_cfg = None

import atexit
import os
import shutil
import yaml

import debugging.dataloader_debug_utils as debug_helper
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS
from scripts.configurations.synthetic_cfg import SYNTHETIC_PARAM_CLASSIFICATIONS
from instanceseg.utils import parse, script_setup as script_utils
from instanceseg.utils.misc import y_or_n_input
from instanceseg.utils.script_setup import configure
from instanceseg.utils.configs import override_cfg
from scripts.test import keys_to_transfer_from_train_to_test, get_config_options_from_train_config, query_remove_logdir


def main(replacement_dict_for_sys_args=None, check_clean_tree=True):
    if check_clean_tree:
        script_utils.check_clean_work_tree()

    args, cfg_override_args = parse.parse_args_test(replacement_dict_for_sys_args)
    checkpoint_path = args.logdir

    cfg, groundtruth_outdir, images_outdir, predictions_outdir, split, tester, use_existing_results = \
        setup_tester(args, cfg_override_args, checkpoint_path)

    if not use_existing_results:
        predictions_outdir, groundtruth_outdir, images_outdir, scores_outdir = tester.test(
            split=split, predictions_outdir=predictions_outdir, groundtruth_outdir=groundtruth_outdir,
            images_outdir=images_outdir, save_scores=True)

    print('Input logdir: {}'.format(args.logdir))
    print('Problem config file: {}'.format(tester.exporter.instance_problem_path))
    print('Predictions exported to {}'.format(predictions_outdir))
    print('Ground truth exported to {}'.format(groundtruth_outdir))
    print('Scores exported to {}'.format(groundtruth_outdir))
    atexit.unregister(query_remove_logdir)
    return predictions_outdir, groundtruth_outdir, tester, args.logdir


def setup_tester(args, cfg_override_args, checkpoint_path):
    train_config_path = os.path.join(checkpoint_path, 'config.yaml')
    model_checkpoint_path = os.path.join(checkpoint_path, 'model_best.pth.tar')
    assert os.path.exists(checkpoint_path), 'Checkpoint path does not exist: {}'.format(checkpoint_path)
    assert os.path.exists(model_checkpoint_path), 'Model checkpoint path does not exist: {}'.format(
        model_checkpoint_path)
    assert os.path.exists(train_config_path), 'Config file does not exist: {}'.format(train_config_path)
    cfg = get_config_options_from_train_config(train_config_path=train_config_path, test_split=args.test_split)
    cfg['train_batch_size'] = cfg['{}_batch_size'.format(args.test_split)]
    override_cfg(cfg, cfg_override_args)
    _, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                        config_idx=args.config,
                                        sampler_name=args.sampler,
                                        script_py_file=__file__,
                                        cfg_override_args=cfg_override_args,
                                        additional_logdir_tag='__test_split-{}'.format(args.test_split))
    atexit.register(query_remove_logdir, out_dir)
    split = args.test_split
    if split not in sampler_cfg:
        if split == 'test':
            print('No sampler configuration for test; using validation configuration instead.')
            sampler_cfg[split] = sampler_cfg['val']
        else:
            raise ValueError('Split {} is not in the sampler config'.format(split))
    with open(os.path.join(out_dir, 'train_logdir.txt'), 'w') as f:
        f.write(checkpoint_path)
    # out_dir = checkpoint_path.rstrip('/') + '_test'
    use_existing_results = False
    predictions_outdir, groundtruth_outdir = (os.path.join(out_dir, s) for s in ('predictions', 'groundtruth'))
    images_outdir = groundtruth_outdir.replace('groundtruth', 'orig_images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        if os.path.exists(predictions_outdir) or os.path.exists(groundtruth_outdir):
            y_or_n = y_or_n_input('I found predictions or gt folder in test directory {}.  Remove for fresh test? '
                                  ''.format(out_dir))
            if y_or_n == 'y':
                shutil.rmtree(out_dir)
            else:
                y_or_n = y_or_n_input('Want to use the existing results from the directory?')
                if y_or_n == 'y':
                    use_existing_results = True
                else:
                    raise Exception('Remove directory {} before proceeding.'.format(out_dir))
    tester = script_utils.setup_test(dataset_type=args.dataset, cfg=cfg, out_dir=out_dir, sampler_cfg=sampler_cfg,
                                     model_checkpoint_path=model_checkpoint_path, gpu=args.gpu, splits=('train', split))
    return cfg, groundtruth_outdir, images_outdir, predictions_outdir, split, tester, use_existing_results


if __name__ == '__main__':
    predictions_outdir, groundtruth_outdir, tester, logdir = main()
