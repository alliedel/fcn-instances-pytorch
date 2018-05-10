import os

import torch

from scripts.configurations import voc_cfg
from torchfcn import script_utils
from torchfcn.datasets.voc import ALL_VOC_CLASS_NAMES
from torchfcn.models import model_utils


def build_example_model(**model_cfg_override_kwargs):
    # script_utils.check_clean_work_tree()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    cfg = voc_cfg.default_config
    problem_config = script_utils.get_problem_config(ALL_VOC_CLASS_NAMES, 2)
    for k, v in model_cfg_override_kwargs.items():
        cfg[k] = v
    model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                 checkpoint=None, semantic_init=None, cuda=cuda)
    return model


def test_vgg_freeze():
    model = build_example_model(map_to_semantic=True)
    model_utils.freeze_vgg_module_subset(model)

    frozen_modules, unfrozen_modules = [], []
    for module_name, module in model.named_children():
        module_frozen = all([p.requires_grad is False for p in module.parameters()])
        if module_frozen:
            frozen_modules.append(module_name)
        else:
            assert all([p.requires_grad is True for p in module.parameters()])
            unfrozen_modules.append(module_name)

    non_vgg_frozen_modules = [module_name for module_name in frozen_modules
                          if module_name not in model_utils.VGG_CHILDREN_NAMES]
    vgg_frozen_modules = [module_name for module_name in frozen_modules
                              if module_name in model_utils.VGG_CHILDREN_NAMES]
    for module_name, module in model.named_children():
        if module_name in model_utils.VGG_CHILDREN_NAMES:
            assert all([p.requires_grad is False for p in module.parameters()])
    print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))

    print('VGG modules frozen: {}'.format(vgg_frozen_modules))
    print('Non-VGG modules frozen: {}'.format(non_vgg_frozen_modules))
    print('Modules unfrozen: {}'.format(unfrozen_modules))
    assert set(unfrozen_modules + vgg_frozen_modules + non_vgg_frozen_modules) == \
           set([module[0] for module in model.named_children()])
    assert len([module[0] for module in model.named_children()]) == \
           len(unfrozen_modules + vgg_frozen_modules + non_vgg_frozen_modules)


def test_all():
    test_vgg_freeze()


if __name__ == '__main__':
    test_all()
