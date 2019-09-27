import atexit
import os.path as osp
import sys

from instanceseg.utils.parse import get_override_cfg
from instanceseg.utils.script_setup import setup_train, configure

here = osp.dirname(osp.abspath(__file__))


def query_remove_logdir(logdir):
    from instanceseg.utils import misc
    import os
    import shutil
    if misc.y_or_n_input('Remove {}?'.format(logdir), default='n') == 'y':
        if os.path.exists(logdir):
            shutil.rmtree(logdir)


def get_single_img_data(dataloader, idx=0):
    img, sem_lbl, inst_lbl = None, None, None
    for i, datapoint in enumerate(dataloader):
        if i == idx:
            img = datapoint['image']
            sem_lbl, inst_lbl = datapoint['sem_lbl'], datapoint['inst_lbl']

    return img, (sem_lbl, inst_lbl)


def main():
    # args, cfg_override_args = local_setup(loss)

    gpu = [1]
    dataset = 'cityscapes'
    config_idx = 'overfit_10'

    override_cfg = get_override_cfg(sys.argv[1:], dataset, sampler=None)
    cfg, out_dir, sampler_cfg = configure(dataset_name=dataset, config_idx=config_idx,
                                          sampler_name=None,
                                          script_py_file=__file__,
                                          cfg_override_args=override_cfg)
    atexit.register(query_remove_logdir, out_dir)
    trainer = setup_train(dataset_type=dataset, cfg=cfg, out_dir=out_dir, sampler_cfg=sampler_cfg,
                          gpu=gpu, checkpoint_path=None)

    img, (sem_lbl, inst_lbl) = get_single_img_data(trainer.dataloaders['train'], 0)

    trainer.train()
    train_loss, train_metrics, _ = trainer.validate_split('train')

    img, (sem_lbl, inst_lbl) = get_single_img_data(trainer.dataloaders['train'], 0)

    atexit.unregister(query_remove_logdir)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
