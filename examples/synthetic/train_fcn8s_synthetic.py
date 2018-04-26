#!/usr/bin/env python

import argparse
import numpy as np
import os
import os.path as osp
import tqdm

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import torchfcn
import torchfcn.datasets.voc
import torchfcn.datasets.synthetic
from torchfcn import script_utils, instance_utils, visualization_utils
import skimage.io

default_config = dict(
    max_iteration=100000,
    lr=1.0e-12,
    momentum=0.99,
    weight_decay=0.0005,
    interval_validate=4000,
    matching=True,
    semantic_only_labels=False,
    n_instances_per_class=None,
    set_extras_to_void=True,
    semantic_subset=None,
    filter_images_by_semantic_subset=False,
    optim='sgd',
    single_instance=False,  # map_to_single_instance_problem
    initialize_from_semantic=False,
    bottleneck_channel_capacity=None,
    size_average=True,
    score_multiplier=None,
)

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(  # vanilla
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        size_average=False
    ),
    1: dict(  # 'semantic': mapping all semantic into a single instance
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        single_instance=True,
        size_average=False
    ),
    2: dict(  # instance seg. with initialization from semantic
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        initialize_from_semantic=True,
        bottleneck_channel_capacity='semantic',
        size_average=False,
        score_multiplier=1e-1,
    ),
    3: dict(  # instance seg. with S channels in the bottleneck layers
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity='semantic',
        size_average=False
    ),
    4: dict(  # instance seg. with semantic init. and N channels in the bottleneck layers
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity=None,
        initialize_from_semantic=True,
        size_average=False
    ),
    5: dict(  # instance seg. with an extra instance channel and semantic init.
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity='semantic',
        initialize_from_semantic=True,
        n_instances_per_class=3,
        size_average=False
    ),
}

here = osp.dirname(osp.abspath(__file__))


def main():
    script_utils.check_clean_work_tree()
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=0,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    parser.add_argument('--image_index', type=int, help='Image index to use for train/validation set', default=None)
    parser.add_argument('--semantic-init', help='Checkpoint path of semantic model (e.g. - '
                                                '\'~/data/models/pytorch/semantic_synthetic.pth\'', default=None)
    args = parser.parse_args()
    gpu = args.gpu
    config_idx = args.config

    synthetic_generator_n_instances_per_semantic_id = 2

    cfg = script_utils.create_config_from_default(configurations[config_idx], default_config)
    if args.image_index is not None:
        cfg['image_index'] = args.image_index

    out = script_utils.get_log_dir(osp.basename(__file__).replace(
        '.py', ''), config_idx, script_utils.create_config_copy(cfg),
        parent_directory=osp.dirname(osp.abspath(__file__)))

    print('logdir: {}'.format(out))
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    dataset_kwargs = dict(transform=True, n_max_per_class=synthetic_generator_n_instances_per_semantic_id,
                          map_to_single_instance_problem=cfg['single_instance'])
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    try:
        img, (sl, il) = train_dataset[0]
    except:
        import ipdb; ipdb.set_trace()
        raise Exception('Cannot load an image from your dataset')
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset.copy(modified_length=3), batch_size=1,
                                                       shuffle=False, **loader_kwargs)
    # 0. Problem setup (instance segmentation definition)
    class_names = val_dataset.class_names
    n_semantic_classes = len(class_names)
    n_instances_per_class = cfg['n_instances_per_class'] or \
                            (1 if cfg['single_instance'] else synthetic_generator_n_instances_per_semantic_id)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for sem_cls in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id)
    problem_config.set_class_names(class_names)

    # 2. model

    model = torchfcn.models.FCN8sInstanceAtOnce(
        semantic_instance_class_list=problem_config.semantic_instance_class_list,
        map_to_semantic=False, include_instance_channel0=False,
        bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'])
    print('Number of classes in model: {}'.format(model.n_classes))
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    elif cfg['initialize_from_semantic']:
        semantic_init_path = os.path.expanduser(args.semantic_init)
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
    if cuda:
        model = model.cuda()

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': script_utils.get_parameters(model, bias=False)},
                {'params': script_utils.get_parameters(model, bias=True),
                 #            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                 #                model, bias=False))},
                 #            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                 #                model, bias=True)),

                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    writer = SummaryWriter(log_dir=out)
    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        train_loader_for_val=train_loader_for_val,
        instance_problem=problem_config,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        tensorboard_writer=writer,
        matching_loss=cfg['matching'],
        loader_semantic_lbl_only=cfg['semantic_only_labels'],
        size_average=cfg['size_average']
    )
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
    if metrics[2] < 85:
        print(script_utils.bcolors.FAIL + 'Test FAILED.  mIOU: {}'.format(metrics[2]) + script_utils.bcolors.ENDC)
    else:
        print(script_utils.bcolors.OKGREEN + 'TEST PASSED! mIOU: {}'.format(metrics[2]) + script_utils.bcolors.ENDC)


if __name__ == '__main__':
    main()
