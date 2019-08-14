import atexit

import os
import yaml
from local_pyutils import get_log_dir

import debugging.dataloader_debug_utils as debug_helper
from instanceseg.utils import parse, script_setup as script_utils, misc, configs
from instanceseg.utils.configs import override_cfg
from scripts.configurations import sampler_cfg_registry
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS
from scripts.configurations.synthetic_cfg import SYNTHETIC_PARAM_CLASSIFICATIONS


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
    for k in PARAM_CLASSIFICATIONS.test:
        keys_to_transfer.append(k)
    for k in SYNTHETIC_PARAM_CLASSIFICATIONS.data:
        keys_to_transfer.append(k)

    keys_to_transfer.remove('train_batch_size')

    return keys_to_transfer


def get_config_options_from_train_config(train_config_path, test_split):
    train_config = yaml.safe_load(open(train_config_path, 'r'))
    test_config_options = {
        k: v for k, v in train_config.items() if k in keys_to_transfer_from_train_to_test()
    }
    if '{}_batch_size'.format(test_split) in test_config_options.keys() \
            and test_config_options['{}_batch_size'.format(test_split)] is not None:
        pass
    else:
        if 'val_batch_size' in test_config_options.keys():
            test_config_options['{}_batch_size'.format(test_split)] = train_config['val_batch_size']
        else:
            print('WARNING: validation batch size not specified (probably from an old log).  Using batch size 1.')
            test_config_options['{}_batch_size'.format(test_split)] = 1
    test_config_options['test_split'] = test_split
    return test_config_options


def query_remove_logdir(logdir):
    from instanceseg.utils import misc
    import shutil
    if misc.y_or_n_input('Remove {}?'.format(logdir), default='n') == 'y':
        shutil.rmtree(logdir)


def main(replacement_dict_for_sys_args=None, check_clean_tree=True):
    args, cfg_override_args = parse.parse_args_test(replacement_dict_for_sys_args)

    if args.ignore_git is True:
        check_clean_tree = False

    if check_clean_tree:
        script_utils.check_clean_work_tree()
    else:
        script_utils.check_clean_work_tree(interactive=False)

    checkpoint_path = args.logdir

    model_subpath = 'model_best.pth.tar'
    cfg, groundtruth_outdir, images_outdir, predictions_outdir, split, tester, use_existing_results = \
        setup_tester(args, cfg_override_args, checkpoint_path, model_subpath=model_subpath)

    if cfg['debug_dataloader_only']:
        import tqdm
        for idx, d in tqdm.tqdm(enumerate(tester.dataloaders[args.test_split]), total=len(tester.dataloaders[
                                                                                              args.test_split]),
                                desc='testing dataset loading', leave=True):
            img = d[0]

        debug_helper.debug_dataloader(tester, split=args.test_split)
        atexit.unregister(query_remove_logdir)
        import sys
        sys.exit(0)

    if not use_existing_results:
        predictions_outdir, groundtruth_outdir, images_outdir, scores_outdir = tester.test(
            split=split, predictions_outdir=predictions_outdir, groundtruth_outdir=groundtruth_outdir,
            images_outdir=images_outdir, save_scores=args.save_scores)

    print('Input logdir: {}'.format(args.logdir))
    print('Problem config file: {}'.format(tester.exporter.instance_problem_path))
    print('Predictions exported to {}'.format(predictions_outdir))
    print('Ground truth exported to {}'.format(groundtruth_outdir))
    atexit.unregister(query_remove_logdir)
    return predictions_outdir, groundtruth_outdir, tester, args.logdir


def test_configure(dataset_name, config_idx, sampler_name, script_py_file='unknownscript.py',
                   cfg_override_args=None, parent_script_directory='scripts', additional_logdir_tag=''):
    script_basename = os.path.basename(script_py_file).replace('.py', '')
    parent_directory = os.path.join(script_utils.PROJECT_ROOT, parent_script_directory, 'logs', dataset_name)
    cfg, cfg_to_print = script_utils.get_cfgs(dataset_name=dataset_name, config_idx=config_idx,
                                              cfg_override_args=cfg_override_args)
    if sampler_name is not None or 'sampler' not in cfg:
        cfg['sampler'] = sampler_name
    else:
        sampler_name = cfg['sampler']
    assert cfg['dataset'] == dataset_name, 'Debug Error: cfg[\'dataset\']: {}, ' \
                                           'args.dataset: {}'.format(cfg['dataset'], dataset_name)
    if cfg['dataset_instance_cap'] == 'match_model':
        cfg['dataset_instance_cap'] = cfg['n_instances_per_class']
    sampler_cfg = sampler_cfg_registry.get_sampler_cfg_set(sampler_name)
    out_dir = get_log_dir(os.path.join(parent_directory, script_basename), cfg_to_print,
                          additional_tag=additional_logdir_tag)
    configs.save_config(out_dir, cfg)
    print(misc.color_text('logdir: {}'.format(out_dir), misc.TermColors.OKGREEN))
    return cfg, out_dir, sampler_cfg


def setup_tester(args, cfg_override_args, checkpoint_path, model_subpath='model_best.pth.tar'):
    train_config_path = os.path.join(checkpoint_path, 'config.yaml')
    model_checkpoint_path = os.path.join(checkpoint_path, model_subpath)
    assert os.path.exists(checkpoint_path), 'Checkpoint path does not exist: {}'.format(checkpoint_path)
    assert os.path.exists(model_checkpoint_path), 'Model checkpoint path does not exist: {}'.format(
        model_checkpoint_path)
    assert os.path.exists(train_config_path), 'Config file does not exist: {}'.format(train_config_path)
    cfg = get_config_options_from_train_config(train_config_path=train_config_path, test_split=args.test_split)
    cfg['train_batch_size'] = cfg['{}_batch_size'.format(args.test_split)]
    override_cfg(cfg, cfg_override_args)

    _, out_dir, sampler_cfg = test_configure(dataset_name=args.dataset,
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
    tester = script_utils.setup_test(dataset_type=args.dataset, cfg=cfg, out_dir=out_dir, sampler_cfg=sampler_cfg,
                                     model_checkpoint_path=model_checkpoint_path, gpu=args.gpu, splits=('train', split))
    return cfg, groundtruth_outdir, images_outdir, predictions_outdir, split, tester, use_existing_results


if __name__ == '__main__':
    if os.path.basename(os.path.abspath('.')) == 'debugging' or os.path.basename(os.path.abspath('.')) == 'scripts':
        os.chdir('../')

    predictions_outdir, groundtruth_outdir, tester, logdir = main()
    problem_config_file = tester.exporter.instance_problem_path
    print('Run convert_test_results_to_coco:\n'
          'python scripts/convert_test_results_to_coco.py '
          '--gt_dir {} '
          '--pred_dir {} '
          '--problem_config_file {} '
          '--logdir {}'.format(groundtruth_outdir, predictions_outdir, problem_config_file, logdir))
    # Make sure we can load all the images
