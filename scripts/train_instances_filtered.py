try:
    import argcomplete
except ImportError:
    argcomplete = None
import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
import torch.utils.data

from scripts.configurations import synthetic_cfg, voc_cfg
from scripts.configurations.sampler_cfg import sampler_cfgs
from torchfcn import script_utils, visualization_utils
from torchfcn.models import model_utils

here = osp.dirname(osp.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='dataset', dest='dataset: voc, synthetic')
    dataset_parsers = {
        'voc': subparsers.add_parser('voc', help='VOC dataset options',
                                     epilog='\n\nOverride options:\n' + '\n'.join(
                                         ['--{}: {}'.format(k, v) for k, v in voc_cfg.default_config.items()]),
                                     formatter_class=argparse.RawTextHelpFormatter),
        'synthetic': subparsers.add_parser('synthetic', help='synthetic dataset options')
    }
    for dataset_name, subparser in dataset_parsers.items():
        subparser.add_argument('-c', '--config', type=str, default=0,
                               choices={'synthetic': synthetic_cfg.configurations,
                                        'voc': voc_cfg.configurations}[dataset_name].keys())
        subparser.set_defaults(dataset=dataset_name)
        subparser.add_argument('-g', '--gpu', type=int, required=True)
        subparser.add_argument('--resume', help='Checkpoint path')
        subparser.add_argument('--semantic-init', help='Checkpoint path of semantic model (e.g. - '
                                                       '\'~/data/models/pytorch/semantic_synthetic.pth\'', default=None)
        subparser.add_argument('--single-image-index', type=int, help='Image index to use for train/validation set',
                               default=None)
        subparser.add_argument('--sampler', type=str, choices=sampler_cfgs.keys(), default='default',
                               help='Sampler for dataset')

    if argcomplete:
        argcomplete.autocomplete(parser)
    args, argv = parser.parse_known_args()

    # Config override parser
    cfg_default = {'synthetic': synthetic_cfg.default_config,
                   'voc': voc_cfg.default_config}[args.dataset]
    cfg_override_parser = argparse.ArgumentParser()

    for arg, default_val in cfg_default.items():
        if default_val is not None:
            cfg_override_parser.add_argument('--' + arg, type=type(default_val), default=default_val,
                                             help='cfg override (only recommended for one-off experiments '
                                                  '- set cfg instead)')
        else:
            cfg_override_parser.add_argument('--' + arg, default=default_val,
                                             help='cfg override (only recommended for one-off experiments '
                                                  '- set cfg instead)')

    bad_args = [arg for arg in argv[::2] if arg.replace('-', '') not in cfg_default.keys()]
    argv = [args.dataset] + argv
    if len(bad_args) > 0:
        raise cfg_override_parser.error('bad_args: {}'.format(bad_args))

    # Parse with list of options
    override_cfg_args, leftovers = cfg_override_parser.parse_known_args(argv)
    if len(leftovers) != 0:
        raise ValueError('args not recognized: {}'.format(leftovers))
    # apparently this is failing, so I'm going to have to screen this on my own:

    # Remove options from namespace that weren't defined
    key_list = list(override_cfg_args.__dict__.keys())
    for k in key_list:
        if '--' + k not in argv and '-' + k not in argv:
            delattr(override_cfg_args, k)
        else:
            # some exceptions
            if k == 'clip':
                setattr(override_cfg_args, 'clip', None)

    return args, override_cfg_args


def main():
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg_default = {'synthetic': synthetic_cfg.default_config,
                   'voc': voc_cfg.default_config}[args.dataset]
    cfg_options = {'synthetic': synthetic_cfg.configurations,
                   'voc': voc_cfg.configurations}[args.dataset]
    cfg = script_utils.create_config_from_default(cfg_options[config_idx], cfg_default)
    cfg['dataset'] = args.dataset
    cfg['sampler'] = args.sampler
    non_default_options = script_utils.prune_defaults_from_dict(cfg_default, cfg)

    for key, override_val in cfg_override_args.__dict__.items():
        old_val = cfg.pop(key)
        if override_val != old_val:
            print(script_utils.color_text('Overriding value for {}: {} --> {}'.format(key, old_val, override_val),
                                          script_utils.TermColors.WARNING))
        cfg[key] = override_val
        non_default_options[key] = override_val

    print(script_utils.color_text('non-default cfg values: {}'.format(non_default_options),
                                  script_utils.TermColors.OKBLUE))
    cfg_to_print = non_default_options
    cfg_to_print = script_utils.create_config_copy(cfg_to_print)
    cfg_to_print = script_utils.make_ordered_cfg(cfg_to_print)

    out_dir = script_utils.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                       cfg_to_print,
                                       parent_directory=os.path.join(here, 'logs', args.dataset))
    script_utils.save_config(out_dir, cfg)
    print(script_utils.color_text('logdir: {}'.format(out_dir), script_utils.TermColors.OKGREEN))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()

    np.random.seed(1234)
    torch.manual_seed(1337)
    if args.cuda:
        torch.cuda.manual_seed(1337)
    print('Getting dataloaders...')
    sampler_cfg = sampler_cfgs[args.sampler]
    if sampler_cfg['train_for_val'] is None:
        sampler_cfg['train_for_val'] = sampler_cfgs['default']['train_for_val']

    dataloaders = script_utils.get_dataloaders(cfg, args.dataset, args.cuda, sampler_cfg)
    print('Done getting dataloaders')
    try:
        i, [sl, il] = [d for i, d in enumerate(dataloaders['train']) if i == 0][0]
    except:
        raise
    synthetic_generator_n_instances_per_semantic_id = 2
    n_instances_per_class = cfg['n_instances_per_class'] or \
                            (1 if cfg['single_instance'] else synthetic_generator_n_instances_per_semantic_id)

    # reduce dataloaders to semantic subset before running / generating problem config:
    for key, dataloader in dataloaders.items():
        dataloader.dataset.reduce_to_semantic_subset(cfg['semantic_subset'])
        dataloader.dataset.set_instance_cap(n_instances_per_class)
    problem_config = script_utils.get_problem_config(dataloaders['val'].dataset.class_names, n_instances_per_class,
                                                     map_to_semantic=cfg['map_to_semantic'])

    if args.resume:
        checkpoint = torch.load(args.resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config, args.resume, args.semantic_init,
                                                                 args.cuda)

    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = script_utils.get_optimizer(cfg, model, checkpoint)

    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p.requires_grad is False for p in module.parameters()])
        print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))

    trainer = script_utils.get_trainer(cfg, args.cuda, model, optim, dataloaders, problem_config, out_dir)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

    print('Evaluating final model')
    metrics, (segmentation_visualizations, score_visualizations) = trainer.validate(should_export_visualizations=False)
    viz = visualization_utils.get_tile_image(segmentation_visualizations)
    skimage.io.imsave(os.path.join(here, 'viz_evaluate.png'), viz)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
        Accuracy: {0}
        Accuracy Class: {1}
        Mean IU: {2}
        FWAV Accuracy: {3}'''.format(*metrics))


if __name__ == '__main__':
    main()
