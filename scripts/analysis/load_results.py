import argparse
import os.path as osp

import instanceseg.losses.loss
import instanceseg.losses.match
import instanceseg.utils.configs
import instanceseg.utils.logs
import instanceseg.factory.models
import torch
from instanceseg.models import model_utils
from instanceseg.utils.misc import mkdir_if_needed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def main_check_freeze():
    # Checking which modules were actually learned
    matching_modules, unmatching_modules = model_utils.compare_model_states(initial_model, model)
    init_logdir = '/tmp/scrap_logdir'
    mkdir_if_needed(init_logdir)
    torch.save({
        'epoch': 0,
        'iteration': 0,
        'arch': initial_model.__class__.__name__,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': initial_model.state_dict(),
        'best_mean_iu': 0,
    }, osp.join(init_logdir, 'model_best.pth.tar'))
    instanceseg.utils.configs.save_config(init_logdir, cfg)
    print('matching:')
    print(matching_modules)
    print('non-matching:')
    print(unmatching_modules)


def main_check_cost_matrix():
    # Checking the cost matrix for the first image
    import torch.nn.functional as F
    from instanceseg.losses import xentropy
    data0 = [x for i, x in enumerate(dataloaders['train_for_val']) if i == 0][0]
    img = torch.autograd.Variable(data0[0].cuda())
    sem_lbl = torch.autograd.Variable(data0[1][0].cuda())
    inst_lbl = torch.autograd.Variable(data0[1][1].cuda())
    scores = model.forward(img)
    log_predictions = F.log_softmax(scores, dim=1)
    size_average = True
    semantic_instance_labels = my_trainer.instance_problem.semantic_instance_class_list
    instance_id_labels = my_trainer.instance_problem.instance_count_id_list

    sem_val = 1

    normalizer = (inst_lbl >= 0).float().data.sum()
    num_inst_classes = len(semantic_instance_labels)
    idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == sem_val)]
    try:
        loss_type = my_trainer.loss_type
    except:
        print('Warning: loss type not defined in trainer; assuming cross_entropy for backwards compatibility')
        loss_type = 'cross_entropy'
    cost_list_2d = instanceseg.losses.match.create_pytorch_cost_matrix(
        instanceseg.losses.loss.single_class_component_loss_functions[loss_type],
        my_trainer.log_predictions[0, ...], sem_lbl[0, ...], inst_lbl[0, ...], semantic_instance_labels,
        instance_id_labels, sem_val, size_average=size_average)
    cost_matrix, multiplier = instanceseg.losses.match.convert_pytorch_costs_to_ints(cost_list_2d)

    for ground_truth in range(len(cost_matrix)):
        for prediction in range(len(cost_matrix[0])):
            print('cost_list_2d gt={}, pred={}: {}'.format(ground_truth, prediction, cost_list_2d[prediction][
                ground_truth].data.cpu().numpy().item()))

    for ground_truth in range(len(cost_matrix)):
        for prediction in range(len(cost_matrix[0])):
            print('cost_matrix gt={}, pred={}: {}'.format(ground_truth, prediction, cost_matrix[prediction][
                ground_truth]))

    pred_permutations, loss, loss_components = instanceseg.losses.loss.cross_entropy2d(scores, sem_lbl, inst_lbl,
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


if __name__ == '__main__':
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        instanceseg.utils.logs.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    cuda = torch.cuda.is_available()
    initial_model, start_epoch, start_iteration = instanceseg.factory.models.get_model(cfg, problem_config,
                                                                                       checkpoint_file=None,
                                                                                       semantic_init=None,
                                                                                       cuda=cuda)
    import ipdb; ipdb.set_trace()
    # main_check_freeze()
    # main_check_cost_matrix()
