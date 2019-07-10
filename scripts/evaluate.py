import os
import os.path as osp
import shutil
import yaml

import debugging.helpers as debug_helper
import instanceseg.utils.script_setup as script_utils
from instanceseg.utils import parse
from instanceseg.utils.misc import y_or_n_input
from instanceseg.utils.script_setup import configure
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS
from scripts.configurations.synthetic_cfg import SYNTHETIC_PARAM_CLASSIFICATIONS

here = osp.dirname(osp.abspath(__file__))


def keys_to_transfer_from_train_to_test():
    keys_to_transfer = []
    for k in PARAM_CLASSIFICATIONS.data:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.debug:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.problem_config:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.model:
        keys_to_transfer.append(k)
    for k in PARAM_CLASSIFICATIONS.export:
        keys_to_transfer.append(k)
    for k in SYNTHETIC_PARAM_CLASSIFICATIONS.data:
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


def parse_args(replacement_dict_for_sys_args=None):
    args, cfg_override_args = parse.parse_args_train(replacement_dict_for_sys_args)
    return args, cfg_override_args


def main(replacement_dict_for_sys_args=None):
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args(replacement_dict_for_sys_args)
    checkpoint_path = args.resume
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args)
    if 'test' not in sampler_cfg:
        print('No sampler configuration for test; using validation configuration instead.')
        sampler_cfg['test'] = sampler_cfg['val']
    train_config_path = os.path.join(checkpoint_path, 'config.yaml')
    model_checkpoint_path = os.path.join(checkpoint_path, 'model_best.pth.tar')
    assert os.path.exists(train_config_path)
    assert os.path.exists(model_checkpoint_path)
    cfg = get_config_options_from_train_config(train_config_path=train_config_path)
    out_dir = checkpoint_path.rstrip('/') + '_test'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        y_or_n = y_or_n_input('Remove test directory {}? '.format(out_dir))
        if y_or_n == 'y':
            shutil.rmtree(out_dir)
        else:
            raise Exception('Remove directory {} before proceeding.'.format(out_dir))
    evaluator = script_utils.setup_test(dataset_type=args.dataset, cfg=cfg, out_dir=out_dir, sampler_cfg=sampler_cfg,
                                        model_checkpoint_path=model_checkpoint_path, gpu=args.gpu)

    if cfg['debug_dataloader_only']:
        debug_helper.debug_dataloader(evaluator)
        return
    predictions_outdir, groundtruth_outdir = (os.path.join(out_dir, s) for s in ('predictions', 'groundtruth'))
    print(predictions_outdir, groundtruth_outdir)
    evaluator.test(split='test', predictions_outdir=predictions_outdir, groundtruth_outdir=groundtruth_outdir)
    print('Predictions exported to {}'.format(predictions_outdir))
    print('Ground truth exported to {}'.format(groundtruth_outdir))
    return predictions_outdir, groundtruth_outdir, evaluator


if __name__ == '__main__':
    main()
