import tqdm
from torch.autograd import Variable
import torch
import os.path
import torch.nn.functional as F

import yaml
from torchfcn import script_utils
from torchfcn import instance_utils


class InstanceMetrics(object):
    def __init__(self, problem_config, model, data_loader):
        self.problem_config = problem_config
        self.model = model
        self.data_loader = data_loader

        self.scores = None
        self.assignments = None
        self.softmaxed_scores = None

    def run(self):
        self.scores = self._compile_scores()
        self.assignments = argmax_scores(self.scores)
        self.softmaxed_scores = softmax_scores(self.scores)

    def _compile_scores(self):
        return compile_scores(self.model, self.data_loader)

    def _compute_instances_found_per_sem_cls(self):
        assignments = self.assignments
        raise NotImplementedError
        for channel_idx, (sem_cls, inst_id) in enumerate(zip(self.problem_config.semantic_instance_class_list,
                                                             self.problem_config.instance_count_id_list)):
            assignments[channel_idx, ...]


def compile_scores(model, data_loader):
    training = model.training
    model.eval()
    compiled_scores = None
    n_images = data_loader.batch_size * len(data_loader)
    batch_size = data_loader.batch_size
    for batch_idx, (data, (sem_lbl, inst_lbl)) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader),
            desc='Running dataset through model', ncols=80,
            leave=False):
        data, sem_lbl, inst_lbl = Variable(data, volatile=True), Variable(sem_lbl), Variable(inst_lbl)
        if next(model.parameters()).is_cuda:
            data, sem_lbl, inst_lbl = data.cuda(), sem_lbl.cuda(), inst_lbl.cuda()
        scores = model(data)
        if compiled_scores is None:
            compiled_scores = Variable(torch.zeros(n_images, *list(scores.size())[1:]))
        compiled_scores[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...] = model(data)
    if training:
        model.train()
    return compiled_scores


def softmax_scores(compiled_scores, dim=1):
    return F.softmax(compiled_scores, dim=dim)


def argmax_scores(compiled_scores, dim=1):
    return compiled_scores.max(dim=dim)[1]


def _test():
    logdir = 'scripts/logs/synthetic/TIME-20180430-151222_VCS-b7e0570_MODEL-train_instances_' \
             'CFG-000_F_SEM-False_SSET-None_INIT_SEM-False_SM-None_VOID-True_MA-True_VAL-100_' \
             'DECAY-0.0005_SEM_LS-False_LR-0.0001_1INST-False_ITR-10000_NPER-None_OPTIM-sgd_MO-0.99_BCC-None_SA-True/'
    model_best_path = os.path.join(logdir, 'model_best.pth.tar')
    checkpoint = torch.load(model_best_path)
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    cfg_file = os.path.join(logdir, 'config.yaml')
    cfg = script_utils.create_config_copy(script_utils.load_config(cfg_file), reverse_replacements=True)
    synthetic_generator_n_instances_per_semantic_id = 2
    n_instances_per_class = cfg['n_instances_per_class'] or \
                            (1 if cfg['single_instance'] else synthetic_generator_n_instances_per_semantic_id)

    dataloaders = script_utils.get_dataloaders(cfg, 'synthetic', cuda)
    problem_config = script_utils.get_problem_config(dataloaders['val'].dataset.class_names, n_instances_per_class)
    model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config, checkpoint,
                                                                 semantic_init=None, cuda=cuda)
    compiled_scores = compile_scores(model, dataloaders['val'])
    softmaxed_scores = softmax_scores(compiled_scores)
    assignments = argmax_scores(compiled_scores)
    assert torch.np.allclose(assignments.data.cpu().numpy(), argmax_scores(softmaxed_scores).data.cpu().numpy())

    # test different metrics


if __name__ == '__main__':
    _test()

#
# def compute_analytics(self, sem_label, inst_label, label_preds, pred_scores, pred_permutations):
#     if type(pred_scores) is list:
#         try:
#             pred_scores_stacked = torch.cat(pred_scores, dim=0)
#             abs_scores_stacked = torch.abs(pred_scores_stacked)
#         except:
#             pred_scores_stacked = np.concatenate(pred_scores, axis=0)
#             abs_scores_stacked = np.abs(pred_scores_stacked)
#     else:
#         pred_scores_stacked = pred_scores
#         try:  # if tensor
#             abs_scores_stacked = torch.abs(pred_scores_stacked)
#         except:
#             abs_scores_stacked = np.abs(pred_scores_stacked)
#     try:
#         softmax_scores = F.softmax(pred_scores_stacked, dim=1)
#     except:
#         softmax_scores = F.softmax(torch.from_numpy(pred_scores_stacked), dim=1)
#     analytics = {
#         'scores': {
#             'max': pred_scores_stacked.max(),
#             # 'mean': pred_scores_stacked.mean(),
#             # 'median': pred_scores_stacked.median(),
#             'min': pred_scores_stacked.min(),
#             # 'abs_mean': abs_scores_stacked.mean(),
#             'abs_median': abs_scores_stacked.median(),
#         },
#     }
#     for channel in range(pred_scores_stacked.size(1)):
#         channel_scores = pred_scores_stacked[:, channel, :, :]
#         abs_channel_scores = abs_scores_stacked[:, channel, :, :]
#         analytics['per_channel_scores/{}'.format(channel)] = {
#             'max': channel_scores.max(),
#             # 'mean': channel_scores.mean(),
#             # 'median': channel_scores.median(),
#             'min': channel_scores.min(),
#             # 'abs_mean': abs_channel_scores.mean(),
#             'abs_median': abs_channel_scores.median(),
#         }
#     channel_labels = self.instance_problem.get_channel_labels('{}_{}')
#     for inst_idx, sem_cls in enumerate(self.instance_problem.semantic_instance_class_list):
#         same_sem_cls_channels = [channel for channel, cls in enumerate(
#             self.instance_problem.semantic_instance_class_list) if cls == sem_cls]
#         all_maxes = softmax_scores[:, same_sem_cls_channels, :, :].max(dim=1)[0]
#         all_sums = softmax_scores[:, same_sem_cls_channels, :, :].sum(dim=1)
#         # get 'smearing' level across all channels
#         sem_channel_keyname = 'instance_commitment/{}'.format(self.instance_problem.class_names[sem_cls])
#         if sem_channel_keyname not in analytics.keys():
#             # first of its sem class -- run semantic analytics here
#             relevant_pixels = sem_label == sem_cls
#             import ipdb; ipdb.set_trace()
#             analytics[sem_channel_keyname] = \
#                 (all_maxes[relevant_pixels] / all_sums[relevant_pixels]).mean(),
#
#         # get 'commitment' level within each channel
#         pixels_assigned_to_me = (label_preds == inst_idx)
#         pixels_assigned_to_my_sem_cls = torch.sum([(label_preds == channel) for channel in
#                                                    same_sem_cls_channels])
#         # channel_usage_fraction: fraction of channels of same semantic class that are used
#         inst_channel_keyname = 'channel_usage_fraction/{}'.format(channel_labels[inst_idx])
#         analytics[inst_channel_keyname] = pixels_assigned_to_me.int().sum() / \
#                                           pixels_assigned_to_my_sem_cls.int().sum()
#
#     return analytics
#
