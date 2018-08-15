import torch

from instanceseg.utils.configs import get_parameters


def get_optimizer(cfg, model, checkpoint_file=None):
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': filter(lambda p: False if p is None else p.requires_grad,
                                  get_parameters(model, bias=False))},
                {'params': filter(lambda p: False if p is None else p.requires_grad,
                                  get_parameters(model, bias=True)),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim