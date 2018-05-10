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
from numpy import random

from scripts.configurations import synthetic_cfg, voc_cfg
from torchfcn import script_utils, visualization_utils
from torchfcn.datasets import dataset_statistics
from torchfcn.datasets import samplers

here = osp.dirname(osp.abspath(__file__))

sampler_cfgs = {
    'default': {
        'train':
            {'n_images': None,
             'sem_cls_filter': None,
             'n_instances_range': None,
             },
        'val': 'copy_train',
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None  # Change to reduce amount of images used to 'validate' the training set
            }
    },
    'person_2inst_1img': {
        'train':
            {'n_images': 1,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2inst_2img': {
        'train':
            {'n_images': 2,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2inst_allimg_sameval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
    },
    'person_2inst_20img_sameval': {
        'train':
            {'n_images': 20,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2_4inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, 4),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, 4),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    }
}


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
        subparser.add_argument('--single-image-index', type=int, help='Image index to use for train/validation set',
                               default=None)
        subparser.add_argument('--sampler', type=str, choices=sampler_cfgs.keys(), default='default',
                               help='Sampler for dataset')
    if argcomplete:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def get_sampler(dataset_instance_stats, sequential, sem_cls=None, n_instances_range=None, n_images=None):
    valid_indices = [True for _ in range(len(dataset_instance_stats.dataset))]
    if n_instances_range is not None:
        valid_indices = pairwise_and(valid_indices,
                                     dataset_instance_stats.filter_images_by_n_instances(n_instances_range, sem_cls))
    elif sem_cls is not None:
        valid_indices = pairwise_and(valid_indices, dataset_instance_stats.filter_images_by_semantic_classes(sem_cls))
    if n_images is not None:
        if sum(valid_indices) < n_images:
            raise Exception('Too few images to sample {}.  Choose a smaller value for n_images in the sampler '
                            'config, or change your filtering requirements for the sampler.'.format(n_images))

        # Subsample n_images
        n_images_chosen = 0
        for idx in random.permutation(len(valid_indices)):
            if valid_indices[idx]:
                if n_images_chosen == n_images:
                    valid_indices[idx] = False
                else:
                    n_images_chosen += 1
        try:
            assert sum(valid_indices) == n_images
        except AssertionError:
            import ipdb; ipdb.set_trace()
            raise
    sampler = samplers.sampler_factory(sequential=sequential, bool_index_subset=valid_indices)(
        dataset_instance_stats.dataset)

    return sampler


def get_configured_sampler(dataset_type, dataset, sequential, n_instances_range, n_images, sem_cls_filter,
                           instance_count_file):
    if n_instances_range is not None:
        if dataset_type != 'voc':
            raise NotImplementedError('Need an established place to save instance counts')
        instance_counts = torch.from_numpy(np.load(instance_count_file)) \
            if os.path.isfile(instance_count_file) else None
        stats = dataset_statistics.InstanceDatasetStatistics(dataset, instance_counts)
        if instance_counts is None:
            stats.compute_statistics()
            instance_counts = stats.instance_counts
            np.save(instance_count_file, instance_counts.numpy())
    else:
        stats = dataset_statistics.InstanceDatasetStatistics(dataset)

    my_sampler = get_sampler(stats, sequential=sequential, n_instances_range=n_instances_range, sem_cls=sem_cls_filter,
                             n_images=n_images)
    if n_images:
        assert len(my_sampler.indices) == n_images
    return my_sampler


def get_dataloaders(cfg, dataset_type, cuda, sampler_args):
    # 1. dataset
    if dataset_type == 'synthetic':
        train_dataset, val_dataset = script_utils.get_synthetic_datasets(cfg)
    elif dataset_type == 'voc':
        train_dataset, val_dataset = script_utils.get_voc_datasets(cfg, script_utils.VOC_ROOT)
    else:
        raise ValueError

    # Filter dataset and create dataloaders
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_sampler_cfg = sampler_cfgs[sampler_args]['train']
    sem_cls_filter = train_sampler_cfg.pop('sem_cls_filter', None)
    if sem_cls_filter is not None:
        if isinstance(sem_cls_filter[0], str):
            try:
                sem_cls_filter = [train_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
            except:
                sem_cls_filter = [int(np.where(train_dataset.class_names == class_name)[0][0]) for class_name in \
                                  sem_cls_filter]
    train_instance_count_file = os.path.join(script_utils.VOC_ROOT, 'train_instance_counts.npy')
    train_sampler = get_configured_sampler(dataset_type, train_dataset, sequential=True,
                                           n_instances_range=train_sampler_cfg.pop('n_instances_range', None),
                                           n_images=train_sampler_cfg.pop('n_images', None),
                                           sem_cls_filter=sem_cls_filter,
                                           instance_count_file=train_instance_count_file)
    if train_sampler_cfg:  # Check if there are any keys left
        raise ValueError('I don''t yet know how to process the following keys: {}'.format(
            train_sampler_cfg.keys()))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, **loader_kwargs)

    val_sampler_cfg = sampler_cfgs[sampler_args]['val']
    if isinstance(val_sampler_cfg, str) and val_sampler_cfg == 'copy_train':
        val_sampler = train_loader.sampler.copy(sequential_override=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=val_sampler, **loader_kwargs)
    else:
        sem_cls_filter = val_sampler_cfg.pop('sem_cls_filter', None)
        if sem_cls_filter is not None:
            if isinstance(sem_cls_filter[0], str):
                try:
                    sem_cls_filter = [val_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
                except:
                    sem_cls_filter = [int(np.where(val_dataset.class_names == class_name)[0][0]) for class_name in \
                                      sem_cls_filter]
        val_instance_count_file = os.path.join(script_utils.VOC_ROOT, 'val_instance_counts.npy')
        val_sampler = get_configured_sampler(dataset_type, val_dataset, sequential=True,
                                             n_instances_range=val_sampler_cfg.pop('n_instances_range', None),
                                             n_images=val_sampler_cfg.pop('n_images', None),
                                             sem_cls_filter=sem_cls_filter,
                                             instance_count_file=val_instance_count_file)
        if val_sampler_cfg:  # Check if there are any keys left
            raise ValueError('I don''t yet know how to process the following keys: {}'.format(
                val_sampler_cfg.keys()))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler, **loader_kwargs)
    train_for_val_cfg = sampler_cfgs[sampler_args].pop('train_for_val', None)
    if train_for_val_cfg is None:
        train_for_val_cfg = sampler_cfgs['default']['train_for_val']
    cut_n_images = train_for_val_cfg.pop('n_images', None) or len(train_loader)
    if train_for_val_cfg:  # Check if there are any keys left
        raise ValueError('I don''t yet know how to process the following keys for train_for_val: {}'.format(
            train_for_val_cfg.keys()))

    train_for_val_sampler = train_loader.sampler.copy(sequential_override=True,
                                                      cut_n_images=None if cut_n_images is None
                                                      else min(cut_n_images, len(train_loader)))
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                       sampler=train_for_val_sampler, **loader_kwargs)
    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }


def pairwise_and(list1, list2):
    return [a and b for a, b in zip(list1, list2)]


def pairwise_or(list1, list2):
    return [a or b for a, b in zip(list1, list2)]


def main():
    script_utils.check_clean_work_tree()
    args = parse_args()
    gpu = args.gpu
    config_idx = args.config
    cfg_default = {'synthetic': synthetic_cfg.default_config,
                   'voc': voc_cfg.default_config}[args.dataset]
    cfg_options = {'synthetic': synthetic_cfg.configurations,
                   'voc': voc_cfg.configurations}[args.dataset]
    cfg = script_utils.create_config_from_default(cfg_options[config_idx], cfg_default)
    non_default_options = script_utils.prune_defaults_from_dict(cfg_default, cfg_options[config_idx])
    print('non-default cfg values: {}'.format(non_default_options))
    cfg_to_print = {
        'dataset': args.dataset
    }
    cfg_to_print.update(non_default_options)
    cfg_to_print['sampler'] = args.sampler
    out_dir = script_utils.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                       cfg_to_print,
                                       parent_directory=os.path.join(here, 'logs', args.dataset))
    print('logdir: {}'.format(out_dir))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()

    np.random.seed(1234)
    torch.manual_seed(1337)
    if args.cuda:
        torch.cuda.manual_seed(1337)
    print('Getting dataloaders...')
    dataloaders = get_dataloaders(cfg, args.dataset, args.cuda, args.sampler)
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

    problem_config = script_utils.get_problem_config(dataloaders['val'].dataset.class_names, n_instances_per_class)

    if args.resume:
        checkpoint = torch.load(args.resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config, args.resume, args.semantic_init,
                                                                 args.cuda)

    print('Number of classes in model: {}'.format(model.n_classes))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = script_utils.get_optimizer(cfg, model, checkpoint)

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
