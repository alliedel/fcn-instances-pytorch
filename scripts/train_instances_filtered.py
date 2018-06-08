import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
import torch.utils.data

from scripts.configurations import synthetic_cfg, voc_cfg
from scripts.configurations.sampler_cfg import sampler_cfgs
from torchfcn import script_utils
from torchfcn import visualization_utils
from torchfcn.models import model_utils

here = osp.dirname(osp.abspath(__file__))


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    args, cfg_override_args = script_utils.parse_args()
    return args, cfg_override_args


def get_cfgs(dataset, config_idx, cfg_override_args):
    # dataset = args.dataset
    cfg_default = {'synthetic': synthetic_cfg.get_default_config(),
                   'voc': voc_cfg.get_default_config()}[dataset]
    cfg_options = {'synthetic': synthetic_cfg.configurations,
                   'voc': voc_cfg.configurations}[dataset]
    cfg = script_utils.create_config_from_default(cfg_options[config_idx], cfg_default)
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

    return cfg, cfg_to_print


def main():
    script_utils.check_clean_work_tree()
    args, cfg_override_args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg, cfg_to_print = get_cfgs(dataset=args.dataset, config_idx=config_idx, cfg_override_args=cfg_override_args)
    assert cfg['dataset'] == args.dataset, 'Debug Error: cfg[\'dataset\']: {}, args.dataset: {}'.format(cfg['dataset'],
                                                                                                        args.dataset)
    sampler_cfg = script_utils.get_sampler_cfg(args.sampler)

    out_dir = script_utils.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                       cfg_to_print,
                                       parent_directory=os.path.join(here, 'logs', args.dataset))
    script_utils.save_config(out_dir, cfg)
    print(script_utils.color_text('logdir: {}'.format(out_dir), script_utils.TermColors.OKGREEN))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()

    script_utils.set_random_seeds()

    print('Getting dataloaders...')
    dataloaders = script_utils.get_dataloaders(cfg, args.dataset, args.cuda, sampler_cfg)
    print('Done getting dataloaders')

    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
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
    if not cfg['map_to_semantic']:
        cfg['activation_layers_to_export'] = tuple([x for x in cfg[
            'activation_layers_to_export'] if x is not 'conv1x1_instance_to_semantic'])
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
