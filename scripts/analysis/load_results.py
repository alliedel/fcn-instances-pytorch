import argparse
import os.path as osp
import os
import yaml
from torchfcn import script_utils
from scripts.configurations import voc_cfg
import torch
from torchfcn.models import model_utils
import local_pyutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def load_config(logdir):
    cfg_file = osp.join(logdir, 'config.yaml')
    loaded_cfg_updates = yaml.load(open(cfg_file))
    print(Warning('WARNING: Using legacy code here! Get rid of loading default config.'))
    default_cfg = voc_cfg.default_config

    loaded_cfg = default_cfg
    loaded_cfg.update(loaded_cfg_updates)

    # Handling some poor legacy code
    if 'sset' in loaded_cfg.keys():
        semantic_subset_as_string = loaded_cfg['sset']
        if semantic_subset_as_string == 'personbackground':
            loaded_cfg['semantic_subset'] = ['person', 'background']
        else:
            raise NotImplementedError('legacy code not filled in here')
    return loaded_cfg


def load_logdir(logdir, gpu=0, packed_as_dict=True):
    # logdir: scripts/logs/voc/TIME-20180511-141755_VCS-1a692c3_MODEL-train_instances_filtered_CFG-
    # person_only__freeze_vgg__many_itr_SSET-personbackground_SAMPLER-person_2_4inst_allimg_realval_DATASET-
    # voc_ITR-1000000_VAL-4000'
    cfg = load_config(logdir)
    dataset = cfg['dataset']
    model_pth = osp.join(logdir, 'model_best.pth.tar')
    out_dir = '/tmp'

    problem_config, model, trainer, optim, dataloaders = script_utils.load_everything_from_cfg(
        cfg, gpu, cfg['sampler'], dataset, resume=model_pth, semantic_init=None, out_dir=out_dir)
    if packed_as_dict:
        return dict(cfg=cfg, model_pth=model_pth, out_dir=out_dir, problem_config=problem_config, model=model,
                    trainer=trainer, optim=optim, dataloaders=dataloaders)
    else:
        return cfg, model_pth, out_dir, problem_config, model, trainer, optim, dataloaders


if __name__ == '__main__':
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, trainer, optim, dataloaders = load_logdir(logdir, gpu=args.gpu,
                                                                                              packed_as_dict=False)
    cuda = torch.cuda.is_available()
    initial_model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                         checkpoint=None, semantic_init=None,
                                                                         cuda=cuda)

    matching_modules, unmatching_modules = model_utils.compare_model_states(initial_model, model)
    init_logdir = '/tmp/scrap_logdir'
    local_pyutils.mkdir_if_needed(init_logdir)
    torch.save({
        'epoch': 0,
        'iteration': 0,
        'arch': initial_model.__class__.__name__,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': initial_model.state_dict(),
        'best_mean_iu': 0,
    }, osp.join(init_logdir, 'model_best.pth.tar'))
    script_utils.save_config(init_logdir, cfg)
    print('matching:')
    print(matching_modules)
    print('non-matching:')
    print(unmatching_modules)
    import ipdb; ipdb.set_trace()
