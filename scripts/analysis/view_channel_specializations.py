import argparse
from scripts.analysis import load_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        load_results.load_logdir(logdir, gpu=args.gpu, packed_as_dict=False)

    
