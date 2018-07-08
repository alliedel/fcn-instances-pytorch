import os
import os.path as osp

import local_pyutils
import numpy as np
import skimage.io
import torch
import torch.utils.data

import torchfcn.utils.configs
import torchfcn.utils.logs
import torchfcn.utils.models
import torchfcn.utils.optimizer
import torchfcn.utils.scripts
from torchfcn.utils import configs, data, models, trainers, samplers, scripts
from scripts.configurations import synthetic_cfg, voc_cfg
from torchfcn.analysis import visualization_utils
from torchfcn.models import model_utils

here = osp.dirname(osp.abspath(__file__))


def parse_args():
    args, cfg_override_args = torchfcn.utils.scripts.parse_args()
    return args, cfg_override_args


def get_cfgs(dataset, config_idx, cfg_override_args):
    # dataset = args.dataset
    cfg_default = {'synthetic': synthetic_cfg.get_default_config(),
                   'voc': voc_cfg.get_default_config()}[dataset]
    cfg_options = {'synthetic': synthetic_cfg.configurations,
                   'voc': voc_cfg.configurations}[dataset]
    cfg = torchfcn.utils.configs.create_config_from_default(cfg_options[config_idx], cfg_default)
    non_default_options = torchfcn.utils.scripts.prune_defaults_from_dict(cfg_default, cfg)
    for key, override_val in cfg_override_args.__dict__.items():
        old_val = cfg.pop(key)
        if override_val != old_val:
            print(torchfcn.utils.scripts.color_text('Overriding value for {}: {} --> {}'.format(key, old_val, override_val),
                                                    torchfcn.utils.scripts.TermColors.WARNING))
        cfg[key] = override_val
        non_default_options[key] = override_val

    print(torchfcn.utils.scripts.color_text('non-default cfg values: {}'.format(non_default_options),
                                            scripts.TermColors.OKBLUE))
    cfg_to_print = non_default_options
    cfg_to_print = torchfcn.utils.configs.create_config_copy(cfg_to_print)
    cfg_to_print = configs.make_ordered_cfg(cfg_to_print)

    return cfg, cfg_to_print


def main():
    local_pyutils.check_clean_work_tree()
    args, cfg_override_args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg, cfg_to_print = get_cfgs(dataset=args.dataset, config_idx=config_idx, cfg_override_args=cfg_override_args)
    assert cfg['dataset'] == args.dataset, 'Debug Error: cfg[\'dataset\']: {}, args.dataset: {}'.format(cfg['dataset'],
                                                                                                        args.dataset)
    if cfg['dataset_instance_cap'] == 'match_model':
        cfg['dataset_instance_cap'] = cfg['n_instances_per_class']
    sampler_cfg = samplers.get_sampler_cfg(args.sampler)

    out_dir = torchfcn.utils.logs.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                              cfg_to_print,
                                              parent_directory=os.path.join(here, 'logs', args.dataset))
    torchfcn.utils.configs.save_config(out_dir, cfg)
    print(torchfcn.utils.scripts.color_text('logdir: {}'.format(out_dir), torchfcn.utils.scripts.TermColors.OKGREEN))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()

    torchfcn.utils.scripts.set_random_seeds()

    print('Getting dataloaders...')
    dataloaders = torchfcn.utils.data.get_dataloaders(cfg, args.dataset, args.cuda, sampler_cfg)
    print('Done getting dataloaders')

    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
    problem_config = torchfcn.utils.models.get_problem_config(dataloaders['val'].dataset.semantic_class_names,
                                                              n_instances_per_class,
                                                              map_to_semantic=cfg['map_to_semantic'])

    if args.resume:
        checkpoint = torch.load(args.resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = torchfcn.utils.models.get_model(cfg, problem_config, args.resume,
                                                                          args.semantic_init, args.cuda)

    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = torchfcn.utils.optimizer.get_optimizer(cfg, model, checkpoint)

    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p.requires_grad is False for p in module.parameters()])
        print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))
    if not cfg['map_to_semantic']:
        cfg['activation_layers_to_export'] = tuple([x for x in cfg[
            'activation_layers_to_export'] if x is not 'conv1x1_instance_to_semantic'])
    trainer = torchfcn.utils.trainers.get_trainer(cfg, args.cuda, model, optim, dataloaders, problem_config, out_dir)
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
