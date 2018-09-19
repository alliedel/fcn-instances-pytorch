import os.path as osp
import sys

import instanceseg.utils.configs
import instanceseg.utils.logs
import instanceseg.utils.misc
import instanceseg.utils.scripts
from instanceseg.utils.scripts import setup, configure

here = osp.dirname(osp.abspath(__file__))


def get_single_img_data(dataloader, idx=0):
    img, sem_lbl, inst_lbl = None, None, None
    for i, (img, (sem_lbl, inst_lbl)) in enumerate(dataloader):
        if i != idx:
            continue
    return img, (sem_lbl, inst_lbl)


def local_setup(loss):
    if loss == 'iou':
        resume = 'tests/logs/synthetic/TIME-20180912-101236_VCS-7f69ed0_MODEL-' \
                 'overfit_single_image_CFG-000_SAMPLER-overfit_1_INSTMET-1' \
                 '_ITR-500_LOSS-cross_entropy_N_IMAGES_TRAIN-0_N_IMAGES_VAL-0_SA-0_VAL-10/model_best.pth.tar'
    else:
        resume = None

    if loss == 'iou':
        override_args_list = ['-c', 'test_overfit_1_iou']
    else:
        override_args_list = ['-c', 'test_overfit_1_xent']
    replacement_args_list = \
        instanceseg.utils.scripts.construct_args_list_to_replace_sys(dataset_name='synthetic', gpu=0, resume=resume)
    replacement_args_list += override_args_list
    replacement_args_list += sys.argv[1:]

    args, cfg_override_args = instanceseg.utils.scripts.parse_args(replacement_args_list=replacement_args_list)

    return args, cfg_override_args


def main(loss='cross_entropy'):
    args, cfg_override_args = local_setup(loss)
    args.resume = None
    cfg, out_dir, sampler_cfg = configure(dataset_name=args.dataset,
                                          config_idx=args.config,
                                          sampler_name=args.sampler,
                                          script_py_file=__file__,
                                          cfg_override_args=cfg_override_args,
                                          parent_script_directory=osp.basename(osp.dirname(here)))

    trainer = setup(args.dataset, cfg, out_dir, sampler_cfg, gpu=args.gpu, resume=args.resume,
                    semantic_init=args.semantic_init)

    trainer.train()
    train_loss, train_metrics, _ = trainer.validate_split('train')

    img, (sem_lbl, inst_lbl) = [x for x in trainer.train_loader][0]

    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    main()
