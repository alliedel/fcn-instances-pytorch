import torchfcn.factory.data
import torchfcn.factory.models
from scripts.configurations import voc_cfg
from torchfcn.datasets.voc import ALL_VOC_CLASS_NAMES
from torchfcn.datasets import dataset_generator_registry
import os
import torch


if __name__ == '__main__':
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    cfg = voc_cfg.get_default_config()
    print('Getting datasets')
    train_dataset, val_dataset = dataset_generator_registry.get_dataset('voc', cfg)
    img, (sem_lbl, inst_lbl) = train_dataset[0]

    problem_config = torchfcn.factory.models.get_problem_config(ALL_VOC_CLASS_NAMES, cfg['n_instances_per_class'])
    model, start_epoch, start_iteration = torchfcn.factory.models.get_model(cfg, problem_config,
                                                                            checkpoint_file=None, semantic_init=None, cuda=cuda)
    dataloaders = torchfcn.factory.data.get_dataloaders(cfg, dataset_type='voc', cuda=cuda)
    import ipdb; ipdb.set_trace()
    model.forward(img)
