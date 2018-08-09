import os
import os.path as osp

import torch
import torch.utils.data

import scripts.configurations.sampler_cfg
import torchfcn.factory.data
import torchfcn.factory.models
import torchfcn.factory.optimizer
import torchfcn.factory.samplers
import torchfcn.factory.trainers
import torchfcn.utils.configs
import torchfcn.utils.logs
import torchfcn.utils.misc
import torchfcn.utils.scripts
from torchfcn.utils.configs import get_cfgs

here = osp.dirname(osp.abspath(__file__))


def parse_args():
    args, cfg_override_args = torchfcn.utils.scripts.parse_args()
    return args, cfg_override_args


def main():
    torchfcn.utils.scripts.check_clean_work_tree()
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

    out_dir = torchfcn.utils.logs.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                              cfg_to_print,
                                              parent_directory=os.path.join(here, 'logs', args.dataset))
    torchfcn.utils.configs.save_config(out_dir, cfg)
    print(torchfcn.utils.misc.color_text('logdir: {}'.format(out_dir), torchfcn.utils.misc.TermColors.OKGREEN))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    args.cuda = torch.cuda.is_available()

    torchfcn.utils.scripts.set_random_seeds()

    print('Getting dataloaders...')
    dataloaders = torchfcn.factory.data.get_dataloaders(cfg, args.dataset, args.cuda, sampler_cfg)
    dataloaders_default = torchfcn.factory.data.get_dataloaders(cfg, args.dataset, args.cuda,
                                                                scripts.configurations.sampler_cfg.get_sampler_cfg_set())
    print('Done getting dataloaders')

    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
    problem_config = torchfcn.factory.models.get_problem_config(dataloaders['val'].dataset.semantic_class_names,
                                                                n_instances_per_class,
                                                                map_to_semantic=cfg['map_to_semantic'])

    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    first_index = dataloaders['val'].sampler.indices[7]
    img_ss, (sem_lbl_ss, inst_lbl_ss) = dataloaders['val'].dataset[first_index]
    img, (sem_lbl, inst_lbl) = dataloaders_default['val'].dataset[first_index]
    car_mask1 = sem_lbl == dataloaders_default['val'].dataset.semantic_class_names.index('car')
    car_mask2 = sem_lbl_ss == dataloaders['val'].dataset.semantic_class_names.index('car')
    print('Loaded example images: img_ss, (sem_lbl_ss, inst_lbl_ss) as sampled version and img, (sem_lbl, '
          'inst_lbl) as original versions (if the sampler had been None).')
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
