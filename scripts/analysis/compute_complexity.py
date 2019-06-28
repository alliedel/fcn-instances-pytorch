import os

import torch

import instanceseg.factory.data
import instanceseg.factory.models
import instanceseg.factory.optimizer
import instanceseg.factory.trainers
from instanceseg.datasets import synthetic
from instanceseg.utils import script_setup
from scripts.configurations import synthetic_cfg
from instanceseg.analysis import computational_complexity


def setup():
    cfg_override_kwargs = {}

    script_setup.set_random_seeds()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if isinstance(gpu, int) else ','.join(str(gpu))
    cuda = torch.cuda.is_available()

    cfg = synthetic_cfg.get_default_train_config()
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
        instanceseg.factory.trainers.get_trainer(cfg, cuda, model, dataloaders, problem_config, out_dir='/tmp/',
                                                 optim=optim)
    return trainer


def main():
    trainer = setup()
    trainer.train()
    val_loss, metrics, _ = trainer.validate_split(
        should_export_visualizations=False, split='train')
    print('Training set mean IU: {}'.format(metrics[2]))

    trainer.model = computational_complexity.add_flops_counting_methods(trainer.model)

    trainer.model.start_flops_count()

    loader = iter(trainer.train_loader)

    data = loader.next()

    batch = Variable(data[0].cuda())


if __name__ == '__main__':
    main()
