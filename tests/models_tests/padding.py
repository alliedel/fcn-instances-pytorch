from torchfcn import script_utils
from scripts.configurations import voc_cfg
from torchfcn.datasets.voc import ALL_VOC_CLASS_NAMES
import os
import torch


if __name__ == '__main__':
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    cfg = voc_cfg.default_config
    print('Getting datasets')
    train_dataset, val_dataset = script_utils.get_voc_datasets(cfg, '/home/adelgior/data/datasets/')

    img, (sem_lbl, inst_lbl) = train_dataset[0]

    problem_config = script_utils.get_problem_config(ALL_VOC_CLASS_NAMES, cfg['n_instances_per_class'])
    model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                 checkpoint_file=None, semantic_init=None, cuda=cuda)
    dataloaders = script_utils.get_dataloaders(
    import ipdb; ipdb.set_trace()
    model.forward(img)
