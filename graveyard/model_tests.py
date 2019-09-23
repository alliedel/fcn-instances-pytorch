import os

import torch

import instanceseg.factory.data
import instanceseg.factory.models
from scripts.configurations import generic_cfg
from instanceseg.models import model_utils


def build_example_model(**model_cfg_override_kwargs):
    # script_utils.check_clean_work_tree()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if isinstance(gpu, int) else ','.join(str(gpu))
    cuda = torch.cuda.is_available()

    cfg = generic_cfg.get_default_config()
    for k, v in model_cfg_override_kwargs.items():
        cfg[k] = v
    problem_config = instanceseg.factory.models.get_problem_config(...)
    model, start_epoch, start_iteration = instanceseg.factory.models.get_model(cfg, problem_config,
                                                                               checkpoint_file=None, semantic_init=None, cuda=cuda)
    return model


def test_forward_hook():
    model = build_example_model()
    cfg = generic_cfg.get_default_config()
    print('Getting datasets')
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if isinstance(gpu, int) else ','.join(str(gpu))
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, 'cityscapes', cuda, sampler_cfg=None)

    layer_names = ['conv1x1_instance_to_semantic'] if model.map_to_semantic else []
    layer_names += ['upscore8', 'score_pool4']
    activations = None
    for i, (x, y) in enumerate(dataloaders['train']):
        activations = model.get_activations(torch.autograd.Variable(x.cuda()), layer_names)
        if i >= 2:
            break
    assert set(activations.keys()) == set(layer_names)
    try:
        [activations[k].size() for k in activations.keys()]
    except:
        raise Exception('activations should all be tensors')


def test_vgg_freeze():
    model = build_example_model(map_to_semantic=True)
    # model = build_example_model(map_to_semantic=True)
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

    assert non_vgg_frozen_modules == ['conv1x1_instance_to_semantic'], '{}'.format(non_vgg_frozen_modules)


def test_all():
    test_forward_hook()
    test_vgg_freeze()


if __name__ == '__main__':
    test_all()
