import torch
import tqdm

from torch.autograd import Variable
from torch.utils.data import sampler

from instanceseg.utils.tensors import softmax_scores, argmax_scores, center_crop_to_reduced_size


def is_sequential(my_sampler):
    return isinstance(my_sampler, sampler.SequentialSampler) or my_sampler.sequential


# Disabling this inspection because IntTensor(R, C) gives a warning all over the place.
# noinspection PyArgumentList
class InstanceMetrics(object):
    def __init__(self, data_loader, problem_config, component_loss_function=None,
                 augment_function_img_sem=None, flag_write_channel_utilization=True,
                 flag_write_loss_distributions=True):
        assert component_loss_function is not None, Warning('I think you want to input the losses function.  If not, '
                                                            'get rid of this line.')
        self.problem_config = problem_config
        self.data_loader = data_loader
        self.component_loss_function = component_loss_function
        self.variables_to_preserve = ('problem_config', 'data_loader', 'component_loss_function',
                                      'variables_to_preserve', 'augment_function_img_sem',
                                      'flag_write_channel_utilization')
        self.flag_write_channel_utilization = flag_write_channel_utilization
        self.flag_write_loss_distributions = flag_write_loss_distributions

        assert not isinstance(self.data_loader.sampler, sampler.RandomSampler), \
            'Sampler is instance of RandomSampler. Please set shuffle to False on data_loader'
        assert is_sequential(self.data_loader.sampler), NotImplementedError
        self.scores = None
        self.losses = None
        self.loss_components = None
        self.assignments = None
        self.softmaxed_scores = None
        self.n_pixels_assigned_per_channel = None
        self.n_instances_assigned_per_sem_cls = None
        self.n_found_per_sem_cls, self.n_missed_per_sem_cls, self.channels_of_majority_assignments = None, None, None
        self.metrics_computed = False
        self.augment_function_img_sem = augment_function_img_sem

    def clear(self, variables_to_preserve=None):
        """
        Clear to evaluate a new model
        """
        if variables_to_preserve is None:
            variables_to_preserve = self.variables_to_preserve
        for attr_name in self.__dict__.keys():
            if attr_name in variables_to_preserve:
                continue
            setattr(self, attr_name, None)

    def compute_metrics(self, model):
        compiled_scores, compiled_losses, compiled_loss_components = self._compile_scores_and_losses(model)
        self.assignments = argmax_scores(compiled_scores)
        self.softmaxed_scores = softmax_scores(compiled_scores)
        if self.flag_write_channel_utilization:
            self.n_pixels_assigned_per_channel = self._compute_pixels_assigned_per_channel(self.assignments)
            self.n_instances_assigned_per_sem_cls = \
                self._compute_instances_assigned_per_sem_cls(self.n_pixels_assigned_per_channel)
            self.n_found_per_sem_cls, self.n_missed_per_sem_cls, self.channels_of_majority_assignments = \
                self._compute_majority_assignment_stats(self.assignments)
        self.metrics_computed = True
        self.scores = compiled_scores
        self.losses = compiled_losses
        self.loss_components = compiled_loss_components

    def _compile_scores_and_losses(self, model):
        compiled_scores, compiled_losses, compiled_loss_components = compile_scores_and_losses(
            model, self.data_loader, self.component_loss_function, self.augment_function_img_sem)
        return compiled_scores, compiled_losses, compiled_loss_components

    def _compute_pixels_assigned_per_channel(self, assignments):
        n_images = len(self.data_loader)
        n_inst_classes = len(self.problem_config.semantic_instance_class_list)
        n_pixels_assigned_per_channel = torch.IntTensor(n_images, n_inst_classes).zero_()
        for channel_idx, (sem_cls, inst_id) in enumerate(zip(self.problem_config.semantic_instance_class_list,
                                                             self.problem_config.instance_count_id_list)):
            n_pixels_assigned_per_channel[:, channel_idx] = \
                (assignments[:, :, :] == channel_idx).int().sum(dim=2).sum(dim=1)
        return n_pixels_assigned_per_channel

    def _compute_instances_assigned_per_sem_cls(self, pixels_assigned_per_channel):
        n_images = len(self.data_loader)
        instances_found_per_channel = torch.IntTensor(n_images, self.problem_config.n_semantic_classes).zero_()
        for channel_idx, (sem_cls, inst_id) in enumerate(zip(self.problem_config.semantic_instance_class_list,
                                                             self.problem_config.instance_count_id_list)):
            instances_found_per_channel[:, sem_cls] += (pixels_assigned_per_channel[:, channel_idx] > 0).int()
        return instances_found_per_channel

    def _compute_majority_assignment_stats(self, assignments, majority_fraction=0.5):
        n_images = len(self.data_loader)
        n_found_per_sem_cls = torch.IntTensor(n_images, self.problem_config.n_semantic_classes).zero_()
        n_missed_per_sem_cls = torch.IntTensor(n_images, self.problem_config.n_semantic_classes).zero_()
        channels_of_majority_assignments = \
            torch.IntTensor(n_images, len(self.problem_config.semantic_instance_class_list)).zero_()

        already_matched_pred_channels = torch.ByteTensor(len(self.problem_config.semantic_instance_class_list))
        assumed_image_size = (assignments.size(1), assignments.size(2))
        for data_idx, (_, (sem_lbl, inst_lbl)) in enumerate(self.data_loader):
            already_matched_pred_channels.zero_()
            for gt_channel_idx, (sem_cls, inst_id) in enumerate(zip(self.problem_config.semantic_instance_class_list,
                                                                    self.problem_config.instance_count_id_list)):
                instance_mask = (sem_lbl == sem_cls) * (inst_lbl == inst_id)
                if (instance_mask.size(1), instance_mask.size(2)) != assumed_image_size:
                    instance_mask = center_crop_to_reduced_size(instance_mask, assumed_image_size, rc_axes=(1, 2))

                # Find majority assignment for this gt instance
                if instance_mask.sum() == 0:
                    continue
                instance_assignments = assignments[data_idx, :, :][instance_mask]
                try:
                    mode_as_torch_tensor = torch.mode(instance_assignments)[0]
                except:
                    import ipdb;
                    ipdb.set_trace()
                    raise Exception
                pred_channel_idx = mode_as_torch_tensor.numpy().item()  # mode returns (val, idx)
                if already_matched_pred_channels[pred_channel_idx]:
                    # We've already assigned this channel to an instance; can't double-count.
                    n_missed_per_sem_cls[data_idx, sem_cls] += 1
                else:
                    # Check whether the majority assignment comprises over half of the pixels
                    n_instance_pixels = instance_mask.float().sum()
                    n_pixels_assigned = (instance_assignments == pred_channel_idx).float().sum()
                    if n_pixels_assigned >= n_instance_pixels * majority_fraction:  # is_majority
                        n_found_per_sem_cls[data_idx, sem_cls] += 1
                        channels_of_majority_assignments[data_idx, pred_channel_idx] += 1
                        already_matched_pred_channels[pred_channel_idx] = True
                    else:
                        n_missed_per_sem_cls[data_idx, sem_cls] += 1
                        n_found_per_sem_cls[data_idx, sem_cls] += 1
        return n_found_per_sem_cls, n_missed_per_sem_cls, channels_of_majority_assignments

    def get_aggregated_scalar_metrics_as_nested_dict(self):
        """
        Aggregate metrics over images and return list of metrics to summarize the performance of the model.
        """
        assert self.metrics_computed, 'Run compute_metrics first'
        channel_labels = self.problem_config.get_channel_labels('{}_{}')
        sem_labels = self.problem_config.semantic_class_names
        sz_score_by_sem = list(self.softmaxed_scores.size())
        sz_score_by_sem[1] = self.problem_config.n_semantic_classes
        softmax_scores_per_sem_cls = torch.zeros(tuple(sz_score_by_sem))
        for sem_cls in range(self.problem_config.n_semantic_classes):
            chs = [ci for ci, sc in
                   enumerate(self.problem_config.semantic_instance_class_list) if sc == sem_cls]
            softmax_scores_per_sem_cls[:, sem_cls, ...] = self.softmaxed_scores[:, chs, ...].sum(dim=1)
        metrics_dict = {
            'n_instances_assigned_per_sem_cls':
                {
                    self.problem_config.semantic_class_names[sem_cls] + '_sum': self.n_instances_assigned_per_sem_cls[:,
                                                                                sem_cls].sum()
                    for sem_cls in range(self.n_instances_assigned_per_sem_cls.size(1))
                },
            'n_images_with_more_than_one_instance_assigned':
                {
                    self.problem_config.semantic_class_names[sem_cls] + '_sum':
                        (self.n_instances_assigned_per_sem_cls[:, sem_cls] > 1).sum()
                    for sem_cls in range(self.n_instances_assigned_per_sem_cls.size(1))
                },
            'channel_utilization':
                {
                    'assignment': {
                        'pixels':
                            {
                                channel_labels[channel_idx] + '_mean': torch.mean(
                                    (self.n_pixels_assigned_per_channel.float())[:, channel_idx])
                                for channel_idx in range(self.n_pixels_assigned_per_channel.size(1))
                            },
                        'instances':
                            {
                                channel_labels[channel_idx] + '_sum':
                                    self.channels_of_majority_assignments[:, channel_idx].sum()
                                for channel_idx in range(self.channels_of_majority_assignments.size(1))
                            },
                    },
                    'softmax_score': {
                        'value_for_assigned_pixels': {
                            channel_labels[channel_idx] + '_mean':
                                0 if (self.assignments == channel_idx).sum() == 0
                                else (self.softmaxed_scores[:, channel_idx, :, :][self.assignments ==
                                                                                  channel_idx]).mean()
                            for channel_idx in range(self.softmaxed_scores.size(1))
                        },
                        'fraction_of_sem_cls_for_assigned_pixels': {
                            channel_labels[channel_idx] + '_mean': 0 if (self.assignments == channel_idx).sum() == 0
                            else ((self.softmaxed_scores[:, channel_idx, :, :][self.assignments == channel_idx]) /
                                  softmax_scores_per_sem_cls[:, sem_cls, :, :][self.assignments == channel_idx]).mean()
                            for channel_idx, sem_cls in enumerate(self.problem_config.semantic_instance_class_list)
                        },
                    },
                    'score': {
                        'value_for_assigned_pixels': {
                            channel_labels[channel_idx] + '_max':
                                0 if (self.assignments == channel_idx).sum() == 0
                                else (self.scores[:, channel_idx, :, :][self.assignments == channel_idx]).mean()
                            for channel_idx in range(self.softmaxed_scores.size(1))
                        },
                    },
                },
            'perc_found_per_sem_cls':
                {
                    sem_labels[sem_cls] + '_mean': torch.mean(self.n_found_per_sem_cls[:, sem_cls].float() / (
                            self.n_found_per_sem_cls[:, sem_cls] + self.n_missed_per_sem_cls[:, sem_cls]).float())
                    for sem_cls in range(self.n_instances_assigned_per_sem_cls.size(1))
                },
            'n_found_per_sem_cls':
                {
                    sem_labels[sem_cls] + '_mean': torch.mean(self.n_found_per_sem_cls[:, sem_cls].float())
                    for sem_cls in range(self.n_instances_assigned_per_sem_cls.size(1))
                },
        }
        return metrics_dict

    def get_aggregated_histogram_metrics_as_nested_dict(self):
        """
        Aggregate metrics over images and return list of metrics to summarize the performance of the model.
        """
        if self.flag_write_loss_distributions is False:
            return None

        assert self.metrics_computed, 'Run compute_metrics first'
        channel_labels = self.problem_config.get_channel_labels('{}_{}')
        sem_labels = self.problem_config.semantic_class_names
        sz_score_by_sem = list(self.softmaxed_scores.size())
        sz_score_by_sem[1] = self.problem_config.n_semantic_classes
        softmax_scores_per_sem_cls = torch.zeros(tuple(sz_score_by_sem))
        for sem_cls in range(self.problem_config.n_semantic_classes):
            chs = [ci for ci, sc in
                   enumerate(self.problem_config.semantic_instance_class_list) if sc == sem_cls]
            softmax_scores_per_sem_cls[:, sem_cls, ...] = self.softmaxed_scores[:, chs, ...].sum(dim=1)
        histogram_metrics_dict = {
            'loss_per_image':
                {
                    'total': self.losses
                },
            'loss_per_channel':
                {
                    channel_labels[channel_idx]:
                        self.loss_components[:, channel_idx]
                    for channel_idx in range(self.loss_components.size(1))
                }
        }

        return histogram_metrics_dict

    def get_images_based_on_characteristics(self):
        image_characteristics = {
            'multiple_instances_detected':
                [n_instances > 1 for n_instances in self.n_instances_assigned_per_sem_cls.max(dim=1)[0]],
        }
        return image_characteristics


def get_same_sem_cls_channels(channel_idx, semantic_instance_class_list):
    return [ci for ci, sc in enumerate(semantic_instance_class_list)
            if sc == semantic_instance_class_list[channel_idx]]


def cat_dictionaries(dictionary, dictionary_to_add):
    dictionary.update(dictionary_to_add)
    return dictionary


def compile_scores_and_losses(model, data_loader, component_loss_function,
                              augment_function_img_sem=None):
    """
    component_loss_function: must be of the form loss_function(scores, sem_lbl, inst_lbl)
    """
    assert component_loss_function is not None
    training = model.training
    model.eval()
    n_images = data_loader.batch_size * len(data_loader)
    n_channels = model.n_output_channels
    batch_size = data_loader.batch_size
    min_image_size, max_image_size = (torch.np.inf, torch.np.inf), (0, 0)
    for batch_idx, (img_data, (sem_lbl, inst_lbl)) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc='Running dataset through model', ncols=80,
            leave=False):
        min_image_size = (min(min_image_size[0], img_data.size(2)), min(min_image_size[1], img_data.size(3)))
        max_image_size = (max(max_image_size[0], img_data.size(2)), min(max_image_size[1], img_data.size(3)))

    compiled_scores = torch.ones(n_images, n_channels, *list(min_image_size))
    compiled_losses = torch.ones(n_images) if component_loss_function is not None else None
    compiled_loss_components = torch.ones(n_images, n_channels) if component_loss_function is not None else None
    for batch_idx, (img_data, (sem_lbl, inst_lbl)) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc='Running dataset through model', ncols=80,
            leave=False):
        if next(model.parameters()).is_cuda:
            img_data, sem_lbl, inst_lbl = img_data.cuda(), sem_lbl.cuda(), inst_lbl.cuda()
        if augment_function_img_sem is not None:
            full_data = augment_function_img_sem(img_data, sem_lbl)
        else:
            full_data = img_data
        full_data, sem_lbl, inst_lbl = Variable(full_data, volatile=True), Variable(sem_lbl), Variable(inst_lbl)
        scores = model(full_data)
        try:
            compiled_scores[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...] = scores.data
        except:
            if all([s1 > s2 for s1, s2 in zip(
                    compiled_scores[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...].size(),
                    scores.size())]):
                import ipdb;
                ipdb.set_trace()
                raise
            else:
                # print(Warning('Cropping image from size {} to {} for easier analysis'.format(
                #     scores.size(), compiled_scores.size())))
                cropped_size = (compiled_scores.size(2), compiled_scores.size(3))
                cropped_scores = center_crop_to_reduced_size(scores, cropped_size, rc_axes=(2, 3))
                try:
                    assert cropped_scores.size() == \
                           compiled_scores[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...].size()
                except:
                    import ipdb;
                    ipdb.set_trace()
                    raise
                compiled_scores[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...] = cropped_scores.data
        if component_loss_function is not None:
            pred_permutations_batch, loss_batch, loss_components = component_loss_function(scores, sem_lbl, inst_lbl)
            compiled_loss_components[(batch_idx * batch_size):((batch_idx + 1) * batch_size), :] = loss_components.data
            compiled_losses[(batch_idx * batch_size):((batch_idx + 1) * batch_size)] = loss_batch.data

    if training:
        model.train()
    return compiled_scores, compiled_losses, compiled_loss_components
