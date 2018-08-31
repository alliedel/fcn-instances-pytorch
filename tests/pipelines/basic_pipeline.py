import os

import torch

import instanceseg.factory.data
import instanceseg.factory.models
import instanceseg.factory.optimizer
import instanceseg.factory.trainers
from instanceseg.datasets import synthetic
from instanceseg.utils import scripts
from scripts.configurations import synthetic_cfg


def main():

    cfg_override_kwargs = {}

    scripts.set_random_seeds()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    cfg = synthetic_cfg.get_default_config()
    for k, v in cfg_override_kwargs.items():
        cfg[k] = v
    problem_config = instanceseg.factory.models.get_problem_config(
        synthetic.ALL_BLOB_CLASS_NAMES, n_instances_per_class=cfg['n_instances_per_class'])
    model, start_epoch, start_iteration = instanceseg.factory.models.get_model(
        cfg, problem_config, checkpoint_file=None, semantic_init=None, cuda=cuda)

    print('Getting datasets')
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, 'synthetic', cuda, sampler_cfg=None)

    optim = instanceseg.factory.optimizer.get_optimizer(cfg, model)
    trainer = \
        instanceseg.factory.trainers.get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir='/tmp/')
    trainer.max_iter = 100
    trainer.train()
    evaluation_metrics, (segmentation_visualizations, score_visualizations) = trainer.validate(
        should_export_visualizations=False, split='train')
    print('Training set mean IU: {}'.format(evaluation_metrics[2]))


if __name__ == '__main__':
    main()
