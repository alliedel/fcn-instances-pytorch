import argparse
import os.path as osp
import os
import yaml
from torchfcn import script_utils
from scripts.configurations import voc_cfg
import torch
from torchfcn.models import model_utils
import local_pyutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def load_config(logdir):
    cfg_file = osp.join(logdir, 'config.yaml')
    # loaded_cfg_updates = yaml.load(open(cfg_file))
    # print(Warning('WARNING: Using legacy code here! Get rid of loading default config.'))
    # default_cfg = voc_cfg.default_config
    #
    # loaded_cfg = default_cfg
    # loaded_cfg.update(loaded_cfg_updates)
    loaded_cfg = yaml.load(open(cfg_file))

    # Handling some poor legacy code
    # if 'sset' in loaded_cfg.keys():
    #     semantic_subset_as_string = loaded_cfg['sset']
    #     if semantic_subset_as_string == 'personbackground':
    #         loaded_cfg['semantic_subset'] = ['person', 'background']
    #     else:
    #         raise NotImplementedError('legacy code not filled in here')
    return loaded_cfg


def load_logdir(logdir, gpu=0, packed_as_dict=True):
    # logdir: scripts/logs/voc/TIME-20180511-141755_VCS-1a692c3_MODEL-train_instances_filtered_CFG-
    # person_only__freeze_vgg__many_itr_SSET-personbackground_SAMPLER-person_2_4inst_allimg_realval_DATASET-
    # voc_ITR-1000000_VAL-4000'
    cfg = load_config(logdir)
    try:
        dataset = cfg['dataset']
    except:
        print('WARNING: remove this for legacy code')
        cfg['dataset'] = 'voc'
        dataset = cfg['dataset']
        cfg['sampler'] = 'person_2_4inst_allimg_realval'
    model_pth = osp.join(logdir, 'model_best.pth.tar')
    out_dir = '/tmp'

    problem_config, model, trainer, optim, dataloaders = script_utils.load_everything_from_cfg(
        cfg, gpu, cfg['sampler'], dataset, resume=model_pth, semantic_init=None, out_dir=out_dir)
    if packed_as_dict:
        return dict(cfg=cfg, model_pth=model_pth, out_dir=out_dir, problem_config=problem_config, model=model,
                    trainer=trainer, optim=optim, dataloaders=dataloaders)
    else:
        return cfg, model_pth, out_dir, problem_config, model, trainer, optim, dataloaders


def main_check_freeze():
    # Checking which modules were actually learned
    matching_modules, unmatching_modules = model_utils.compare_model_states(initial_model, model)
    init_logdir = '/tmp/scrap_logdir'
    local_pyutils.mkdir_if_needed(init_logdir)
    torch.save({
        'epoch': 0,
        'iteration': 0,
        'arch': initial_model.__class__.__name__,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': initial_model.state_dict(),
        'best_mean_iu': 0,
    }, osp.join(init_logdir, 'model_best.pth.tar'))
    script_utils.save_config(init_logdir, cfg)
    print('matching:')
    print(matching_modules)
    print('non-matching:')
    print(unmatching_modules)


if __name__ == '__main__':
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = load_logdir(logdir, gpu=args.gpu,
                                                                                                 packed_as_dict=False)
    cuda = torch.cuda.is_available()
    initial_model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                         checkpoint=None, semantic_init=None,
                                                                         cuda=cuda)

    # main_check_freeze()

    # Checking the cost matrix for the first image
    import torch.nn.functional as F
    from torchfcn import losses
    data0 = [x for i, x in enumerate(dataloaders['train_for_val']) if i == 0][0]
    img = torch.autograd.Variable(data0[0].cuda())
    sem_lbl = torch.autograd.Variable(data0[1][0].cuda())
    inst_lbl = torch.autograd.Variable(data0[1][1].cuda())
    scores = model.forward(img)
    log_predictions = F.log_softmax(scores, dim=1)
    size_average=True
    semantic_instance_labels = my_trainer.instance_problem.semantic_instance_class_list
    instance_id_labels = my_trainer.instance_problem.instance_count_id_list

    sem_val = 1

    normalizer = (inst_lbl >= 0).float().data.sum()
    num_inst_classes = len(semantic_instance_labels)
    idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == sem_val)]
    cost_list_2d = losses.create_pytorch_cross_entropy_cost_matrix(log_predictions[0, ...], sem_lbl[0, ...],
                                                                   inst_lbl[0, ...],
                                                                   semantic_instance_labels,
                                                                   instance_id_labels,
                                                                   sem_val, size_average=size_average)
    cost_matrix, multiplier = losses.convert_pytorch_costs_to_ints(cost_list_2d)

    for ground_truth in range(len(cost_matrix)):
        for prediction in range(len(cost_matrix[0])):
            print('cost_list_2d gt={}, pred={}: {}'.format(ground_truth, prediction, cost_list_2d[prediction][
                ground_truth].data.cpu().numpy().item()))

    for ground_truth in range(len(cost_matrix)):
        for prediction in range(len(cost_matrix[0])):
            print('cost_matrix gt={}, pred={}: {}'.format(ground_truth, prediction, cost_matrix[prediction][
                ground_truth]))

    pred_permutations, loss, loss_components = losses.cross_entropy2d(scores, sem_lbl, inst_lbl,
                                                                      semantic_instance_labels,
                                                                      instance_id_labels, matching=True,
                                                                      break_here=False, recompute_optimal_loss=False,
                                                                      return_loss_components=True,
                                                                      size_average=size_average)
    inst_lbl_pred = scores.data.max(1)[1].cpu().numpy()[:, :, :]
    lt_combined = my_trainer.gt_tuple_to_combined(sem_lbl.data.cpu().numpy(), inst_lbl.data.cpu().numpy())
    metrics = my_trainer.compute_metrics(label_trues=lt_combined, label_preds=inst_lbl_pred,
                                         permutations=pred_permutations, single_batch=True)

    import ipdb; ipdb.set_trace()
