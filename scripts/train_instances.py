import argparse
import numpy as np
import os
import os.path as osp

import torch
from tensorboardX import SummaryWriter

import torchfcn
import torchfcn.datasets.voc
import torchfcn.datasets.synthetic
from torchfcn import script_utils, instance_utils, visualization_utils
import skimage.io

from scripts.configurations import synthetic_cfg, voc_cfg

here = osp.dirname(osp.abspath(__file__))

VOC_ROOT = osp.expanduser('~/data/datasets')


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='dataset', dest='dataset: voc, synthetic')
    dataset_parsers = {
        'voc': subparsers.add_parser('voc', help='VOC dataset options'),
        'synthetic': subparsers.add_parser('synthetic', help='synthetic dataset options')
    }
    for dataset_name, subparser in dataset_parsers.items():
        subparser.add_argument('-c', '--config', type=int, default=0,
                               choices={'synthetic': synthetic_cfg.configurations,
                                        'voc': voc_cfg.configurations}[dataset_name].keys())
        subparser.set_defaults(dataset=dataset_name)
        subparser.add_argument('-g', '--gpu', type=int, required=True)
        subparser.add_argument('--resume', help='Checkpoint path')
        subparser.add_argument('--semantic-init', help='Checkpoint path of semantic model (e.g. - '
                                                    '\'~/data/models/pytorch/semantic_synthetic.pth\'', default=None)

    args = parser.parse_args()
    return args


def get_trainer(cfg, args, model, optim, dataloaders, problem_config, out_dir):
    writer = SummaryWriter(log_dir=out_dir)
    trainer = torchfcn.Trainer(
        cuda=args.cuda,
        model=model,
        optimizer=optim,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        train_loader_for_val=dataloaders['train_for_val'],
        instance_problem=problem_config,
        out=out_dir,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
        tensorboard_writer=writer,
        matching_loss=cfg['matching'],
        loader_semantic_lbl_only=cfg['semantic_only_labels'],
        size_average=cfg['size_average']
    )
    return trainer


def get_optimizer(cfg, model, checkpoint=None):
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': script_utils.get_parameters(model, bias=False)},
                {'params': script_utils.get_parameters(model, bias=True),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if checkpoint:
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim


# val_dataset.class_names
def get_problem_config(class_names, n_instances_per_class):
    # 0. Problem setup (instance segmentation definition)
    class_names = class_names
    n_semantic_classes = len(class_names)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for sem_cls in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id)
    problem_config.set_class_names(class_names)
    return problem_config


def get_model(cfg, problem_config, checkpoint, semantic_init, cuda):
    model = torchfcn.models.FCN8sInstanceAtOnce(
        semantic_instance_class_list=problem_config.semantic_instance_class_list,
        map_to_semantic=False, include_instance_channel0=False,
        bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'])
    print('Number of classes in model: {}'.format(model.n_classes))
    if cfg['initialize_from_semantic']:
        semantic_init_path = os.path.expanduser(semantic_init)
        if not os.path.exists(semantic_init_path):
            raise ValueError('I could not find the path {}.  Did you set the path using the semantic-init '
                             'flag?'.format(semantic_init_path))
        semantic_model = torchfcn.models.FCN8sInstanceAtOnce(
            semantic_instance_class_list=[1 for _ in range(problem_config.n_semantic_classes)],
            map_to_semantic=False, include_instance_channel0=False)
        print('Copying params from preinitialized semantic model')
        checkpoint = torch.load(semantic_init_path)
        semantic_model.load_state_dict(checkpoint['model_state_dict'])
        model.copy_params_from_semantic_equivalent_of_me(semantic_model)
    else:
        print('Copying params from vgg16')
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch, start_iteration = 0, 0
    if cuda:
        model = model.cuda()
    return model, start_epoch, start_iteration


def get_synthetic_datasets(cfg):
    synthetic_generator_n_instances_per_semantic_id = 2
    dataset_kwargs = dict(transform=True, n_max_per_class=synthetic_generator_n_instances_per_semantic_id,
                          map_to_single_instance_problem=cfg['single_instance'])
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    return train_dataset, val_dataset


def get_voc_datasets(cfg, voc_root):
    dataset_kwargs = dict(transform=True, semantic_only_labels=cfg['semantic_only_labels'],
                          set_extras_to_void=cfg['set_extras_to_void'], semantic_subset=cfg['semantic_subset'],
                          map_to_single_instance_problem=cfg['single_instance'])
    semantic_subset_as_str = cfg['semantic_subset']
    if semantic_subset_as_str is not None:
        semantic_subset_as_str = '_'.join(cfg['semantic_subset'])
    else:
        semantic_subset_as_str = cfg['semantic_subset']
    instance_counts_cfg_str = '_semantic_subset-{}'.format(semantic_subset_as_str)
    instance_counts_file = osp.expanduser('~/data/datasets/VOC/instance_counts{}.npy'.format(instance_counts_cfg_str))
    if os.path.exists(instance_counts_file):
        print('Loading precomputed instance counts from {}'.format(instance_counts_file))
        instance_precomputed = True
        instance_counts = np.load(instance_counts_file)
        if len(instance_counts.shape) == 0:
            raise Exception('instance counts file contained empty array. Delete it: {}'.format(instance_counts_file))
    else:
        print('No precomputed instance counts (checked in {})'.format(instance_counts_file))
        instance_precomputed = False
        instance_counts = None
    train_dataset_kwargs = dict(weight_by_instance=cfg['weight_by_instance'],
                                instance_counts_precomputed=instance_counts)
    train_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='train', **dataset_kwargs,
                                                          **train_dataset_kwargs)
    if not instance_precomputed:
        try:
            assert train_dataset.instance_counts is not None
            np.save(instance_counts_file, train_dataset.instance_counts)
        except:
            import ipdb; ipdb.set_trace()  # to save from rage-quitting after having just computed the instance counts
            raise
    val_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_dataloaders(cfg, args):
    # 1. dataset
    if args.dataset == 'synthetic':
        train_dataset, val_dataset = get_synthetic_datasets(cfg)
    elif args.dataset == 'voc':
        train_dataset, val_dataset = get_voc_datasets(cfg, VOC_ROOT)
    else:
        raise ValueError
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset.copy(modified_length=3), batch_size=1,
                                                       shuffle=False, **loader_kwargs)
    try:
        img, (sl, il) = train_dataset[0]
    except:
        import ipdb; ipdb.set_trace()
        raise Exception('Cannot load an image from your dataset')

    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }


def main():
    script_utils.check_clean_work_tree()
    args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg_default = {'synthetic': synthetic_cfg.default_config,
                   'voc': voc_cfg.default_config}[args.dataset]
    cfg_options = {'synthetic': synthetic_cfg.configurations,
                   'voc': voc_cfg.configurations}[args.dataset]
    import ipdb; ipdb.set_trace()
    cfg = script_utils.create_config_from_default(cfg_options[config_idx], cfg_default)
    non_default_options = script_utils.prune_defaults_from_dict(cfg_default, cfg_options)
    print('non-default cfg values: {}'.format(non_default_options))
    out_dir = script_utils.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                       script_utils.create_config_copy(cfg),
                                       parent_directory=osp.dirname(osp.abspath(__file__)))
    print('logdir: {}'.format(out_dir))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if args.cuda:
        assert torch.cuda.is_available()

    torch.manual_seed(1337)
    if args.cuda:
        torch.cuda.manual_seed(1337)

    dataloaders = get_dataloaders(cfg, args)

    synthetic_generator_n_instances_per_semantic_id = 2
    n_instances_per_class = cfg['n_instances_per_class'] or \
                            (1 if cfg['single_instance'] else synthetic_generator_n_instances_per_semantic_id)
    problem_config = get_problem_config(dataloaders['val'].dataset.class_names, n_instances_per_class)

    if args.resume:
        checkpoint = torch.load(args.resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = get_model(cfg, problem_config, args.resume, args.semantic_init, args.cuda)

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = get_optimizer(cfg, model, checkpoint)

    trainer = get_trainer(cfg, args, model, optim, dataloaders, problem_config, out_dir)
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
