import instanceseg.losses.loss
import instanceseg.factory.data
import argparse
import instanceseg.utils.logs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def run_test():
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        instanceseg.utils.logs.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)


if __name__ == '__main__':
    run_test()
