import atexit
import os
import shutil
import yaml

import debugging.helpers as debug_helper
from scripts.configurations.generic_cfg import PARAM_CLASSIFICATIONS
from scripts.configurations.synthetic_cfg import SYNTHETIC_PARAM_CLASSIFICATIONS
from instanceseg.utils import parse, script_setup as script_utils
from instanceseg.utils.misc import y_or_n_input
from instanceseg.utils.script_setup import configure
from instanceseg.utils.configs import override_cfg


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
    if check_clean_tree:
        script_utils.check_clean_work_tree()

    args, cfg_override_args = parse.parse_args_test(replacement_dict_for_sys_args)
    checkpoint_path = args.logdir

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
        predictions_outdir, groundtruth_outdir, images_outdir = tester.test(split=split,
                                                                            predictions_outdir=predictions_outdir,
                                                                            groundtruth_outdir=groundtruth_outdir,
                                                                            images_outdir=images_outdir)

    print('Input logdir: {}'.format(args.logdir))
    print('Problem config file: {}'.format(tester.exporter.instance_problem_path))
    print('Predictions exported to {}'.format(predictions_outdir))
    print('Ground truth exported to {}'.format(groundtruth_outdir))
    atexit.unregister(query_remove_logdir)
    return predictions_outdir, groundtruth_outdir, tester, args.logdir


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
