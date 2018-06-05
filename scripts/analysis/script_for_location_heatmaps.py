import argparse
import os.path as osp
from torchfcn import script_utils
import torch
from torchfcn.models import model_utils
import local_pyutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def get_per_image_per_channel_heatmaps(cfg, problem_config, model, dataloader):
    pass


if __name__ == '__main__':
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        script_utils.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    cuda = torch.cuda.is_available()
    initial_model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                         checkpoint_file=None, semantic_init=None,
                                                                         cuda=cuda)

    import ipdb; ipdb.set_trace()
    