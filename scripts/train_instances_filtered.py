import os
import os.path as osp

import numpy as np
import skimage.io
import torch
import torch.utils.data

import scripts.configurations.sampler_cfg
import instanceseg.factory.data
import instanceseg.utils.configs
import instanceseg.utils.logs
import instanceseg.utils.misc
import instanceseg.factory.models
import instanceseg.factory.optimizer
import instanceseg.factory.samplers
import instanceseg.utils.scripts
import instanceseg.factory.trainers
from instanceseg.analysis import visualization_utils
from instanceseg.models import model_utils
from instanceseg.utils.configs import get_cfgs

here = osp.dirname(osp.abspath(__file__))


def main():
    instanceseg.utils.scripts.check_clean_work_tree()
    args, cfg, out_dir, sampler_cfg = configure()
    trainer = setup(args, cfg, out_dir, sampler_cfg)

    print('Evaluating final model')
    metrics = run(trainer)
    print('''\
        Accuracy: {0}
        Accuracy Class: {1}
        Mean IU: {2}
        FWAV Accuracy: {3}'''.format(*metrics))


def setup(args, cfg, out_dir, sampler_cfg):

    print('Getting dataloaders...')
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, args.dataset, args.cuda, sampler_cfg)
    print('Done getting dataloaders')
    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
    problem_config = instanceseg.factory.models.get_problem_config(dataloaders['val'].dataset.semantic_class_names,
                                                                   n_instances_per_class,
                                                                   map_to_semantic=cfg['map_to_semantic'])
    if args.resume:
        checkpoint = torch.load(args.resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = instanceseg.factory.models.get_model(cfg, problem_config, args.resume,
                                                                               args.semantic_init, args.cuda)
    # Run a few checks
    problem_config_semantic_classes = set([problem_config.semantic_class_names[si]
                                           for si in problem_config.semantic_instance_class_list])
    dataset_semantic_classes = set(dataloaders['train'].dataset.semantic_class_names)
    assert problem_config_semantic_classes == dataset_semantic_classes, \
        'Model covers these semantic classes: {}.\n ' \
        'Dataset covers these semantic classes: {}.'.format(problem_config_semantic_classes, dataset_semantic_classes)
    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = instanceseg.factory.optimizer.get_optimizer(cfg, model, checkpoint)
    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p.requires_grad is False for p in module.parameters()])
        print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))
    if not cfg['map_to_semantic']:
        cfg['activation_layers_to_export'] = tuple([x for x in cfg[
            'activation_layers_to_export'] if x is not 'conv1x1_instance_to_semantic'])
    trainer = \
        instanceseg.factory.trainers.get_trainer(cfg, args.cuda, model, optim, dataloaders, problem_config, out_dir)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    return trainer


def run(trainer):
    trainer.train()
    metrics, (segmentation_visualizations, score_visualizations) = trainer.validate(should_export_visualizations=False)
    viz = visualization_utils.get_tile_image(segmentation_visualizations)
    skimage.io.imsave(os.path.join(here, 'viz_evaluate.png'), viz)
    metrics = np.array(metrics)
    metrics *= 100
    return metrics


def configure():
    args, cfg_override_args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg, cfg_to_print = get_cfgs(dataset_name=args.dataset, config_idx=config_idx, cfg_override_args=cfg_override_args)
    cfg['sampler'] = args.sampler
    assert cfg['dataset'] == args.dataset, 'Debug Error: cfg[\'dataset\']: {}, args.dataset: {}'.format(cfg['dataset'],
                                                                                                        args.dataset)
    if cfg['dataset_instance_cap'] == 'match_model':
        cfg['dataset_instance_cap'] = cfg['n_instances_per_class']
    sampler_cfg = scripts.configurations.sampler_cfg.get_sampler_cfg(args.sampler)
    out_dir = instanceseg.utils.logs.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                                 cfg_to_print,
                                                 parent_directory=os.path.join(here, 'logs', args.dataset))
    instanceseg.utils.configs.save_config(out_dir, cfg)
    print(instanceseg.utils.misc.color_text('logdir: {}'.format(out_dir), instanceseg.utils.misc.TermColors.OKGREEN))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()
    instanceseg.utils.scripts.set_random_seeds()
    return args, cfg, out_dir, sampler_cfg


def parse_args():
    args, cfg_override_args = instanceseg.utils.scripts.parse_args()
    return args, cfg_override_args


if __name__ == '__main__':
    main()
