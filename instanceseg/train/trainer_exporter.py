import datetime
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter

from instanceseg.losses.loss import MatchingLossResult
import instanceseg.utils.display as display_pyutils
import instanceseg.utils.export
import instanceseg.utils.imgutils
from instanceseg.analysis import visualization_utils
from instanceseg.datasets import runtime_transformations
from instanceseg.ext.panopticapi.utils import rgb2id, id2rgb
from instanceseg.losses.loss import LossMatchAssignments
from instanceseg.losses.match import GT_VALUE_FOR_FALSE_POSITIVE
from instanceseg.utils import instance_utils
from instanceseg.utils.instance_utils import InstanceProblemConfig
from instanceseg.utils.misc import flatten_dict

display_pyutils.set_my_rc_defaults()

MY_TIMEZONE = 'America/New_York'


def should_write_activations(iteration, epoch):
    if iteration < 3000:
        return True
    else:
        return False


DEBUG_ASSERTS = True


def log(num_or_vec, base):
    # log_b(x) = log_c(x) / log_c(b)
    return np.log(num_or_vec) / np.log(base)


class ConservativeExportDecider(object):
    def __init__(self, base_interval):
        """
        Decides whether we should export something
        """
        self.base_interval = base_interval
        self.n_previous_exports = 0
        self.power = 2  # 3

    def get_export_iteration_list(self, max_iterations):
        return [(x * self.base_interval) ** 3 for x in range(0, np.ceil(log(max_iterations, 3)) + 1)]

    @property
    def next_export_iteration(self):
        return self.get_item_in_sequence(self.n_previous_exports)

    @property
    def current_export_iteration(self):
        return None if self.n_previous_exports == 0 else self.get_item_in_sequence(self.n_previous_exports - 1)

    def get_item_in_sequence(self, index):
        return self.base_interval * (index ** self.power)

    def is_prev_or_next_export_iteration(self, iteration):
        if iteration > self.next_export_iteration:
            raise Exception('Missed an export at iteration {}'.format(self.next_export_iteration ** 3))
        else:
            return iteration == self.next_export_iteration or \
                   self.current_export_iteration is None and iteration == self.current_export_iteration


class ModelHistorySaver(object):
    def __init__(self, model_checkpoint_dir, interval_validate, max_n_saved_models=20, max_n_iterations=100000):
        assert np.mod(max_n_saved_models, 2) == 0, 'Max_n_saved_models must be even'
        self.model_checkpoint_dir = model_checkpoint_dir
        if not os.path.exists(model_checkpoint_dir):
            raise Exception('{} does not exist'.format(model_checkpoint_dir))
        self.interval_validate = interval_validate
        self.max_n_saved_models = max_n_saved_models
        n_digits = max(6, np.ceil(np.log10(max_n_iterations + 1)))
        self.itr_format = '{:0' + str(n_digits) + 'd}'
        self.adaptive_save_model_every = 1

    def get_list_of_checkpoint_files(self):
        return [os.path.join(self.model_checkpoint_dir, f) for f in sorted(os.listdir(self.model_checkpoint_dir))]

    def get_latest_checkpoint_file(self):
        return self.get_list_of_checkpoint_files()[-1]

    def get_model_filename_from_iteration(self, i):
        return os.path.join(self.model_checkpoint_dir, 'model_' + self.itr_format.format(i) + '.pth.tar')

    def get_iteration_from_model_filename(self, model_filename):
        itr_as_06d = os.path.basename(model_filename).split('_')[1].split('.')[0]
        assert itr_as_06d.isdigit()
        return int(itr_as_06d)

    def save_model_to_history(self, current_itr, checkpoint_file_src):
        if np.mod(current_itr, self.adaptive_save_model_every * self.interval_validate) == 0:
            shutil.copyfile(checkpoint_file_src, self.get_model_filename_from_iteration(current_itr))
            self.clean_up_checkpoints()
            return True
        else:
            return False

    def clean_up_checkpoints(self):
        """
        Cleans out history to keep only a small number of models; always ensures we keep the first and most recent.
        """
        most_recent_file = self.get_latest_checkpoint_file()
        most_recent_itr = self.get_iteration_from_model_filename(most_recent_file)
        n_vals_so_far = most_recent_itr / self.interval_validate
        if (n_vals_so_far / self.adaptive_save_model_every) >= (self.max_n_saved_models):
            while (n_vals_so_far / self.adaptive_save_model_every) >= self.max_n_saved_models:
                self.adaptive_save_model_every *= 2  # should use ceil, log2 to compute instead (this is hacky)
            iterations_to_keep = range(0, most_recent_itr + self.interval_validate,
                                       self.adaptive_save_model_every * self.interval_validate)
            if most_recent_itr not in iterations_to_keep:
                iterations_to_keep.append(most_recent_itr)
            for j in iterations_to_keep:  # make sure the files we assume exist actually exist
                assert os.path.exists(self.get_model_filename_from_iteration(j)), \
                    '{} does not exist'.format(f)

            for model_file in self.get_list_of_checkpoint_files():
                iteration_number = self.get_iteration_from_model_filename(model_file)
                if iteration_number not in iterations_to_keep:
                    os.remove(model_file)
            assert len(self.get_list_of_checkpoint_files()) <= (self.max_n_saved_models + 1), 'DebugError'


class ExportConfig(object):
    def __init__(self, interval_validate=None, export_activations=None, activation_layers_to_export=(),
                 write_instance_metrics=False, run_loss_updates=True, max_n_saved_models=None):
        self.interval_validate = interval_validate
        self.export_activations = export_activations
        self.activation_layers_to_export = activation_layers_to_export
        self.write_instance_metrics = write_instance_metrics
        self.run_loss_updates = run_loss_updates

        self.write_activation_condition = should_write_activations
        self.which_heatmaps_to_visualize = 'same semantic'  # 'all'

        self.max_n_saved_models = 20 if max_n_saved_models is None else max_n_saved_models

        self.downsample_multiplier_score_images = 0.5
        self.export_component_losses = True

        self.write_lr = True


def sem_inst_lbls_to_id2rgb(sem_lbl, inst_lbl, void_mapping={255: [255, 255, 255]}):
    im = id2rgb(sem_lbl + 256 * inst_lbl)
    for key, value in void_mapping.items():
        im[sem_lbl == key, :] = value
    return im


def export_inst_sem_lbls_as_id2rgb(sem_lbls_as_batch_nparray, inst_lbls_as_batch_nparray, output_directory,
                                   image_names):
    assert np.sum(sem_lbls_as_batch_nparray < 0) == 0  # semantic labels need to be >= 0 for unique ids
    assert np.sum(inst_lbls_as_batch_nparray < 0) == 0  # instance labels need to be >= 0 for unique ids
    batch_sz = sem_lbls_as_batch_nparray.shape[0]
    assert batch_sz == inst_lbls_as_batch_nparray.shape[0]
    for img_idx in range(batch_sz):
        sem_l = sem_lbls_as_batch_nparray[img_idx, ...]
        inst_l = inst_lbls_as_batch_nparray[img_idx, ...]
        img = sem_inst_lbls_to_id2rgb(sem_l, inst_l)
        img[sem_l == 255, :] = 255  # Convert all void to (255,255,255) instead of, say, (255,0,0)
        out_file = os.path.join(output_directory, image_names[img_idx])
        visualization_utils.write_image(out_file, img)


class TrainerExporter(object):
    log_headers = [
        'epoch',
        'iteration',
        'train/losses',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
        'valid/losses',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
        'elapsed_time',
    ]

    def __init__(self, out_dir, instance_problem: InstanceProblemConfig, export_config: ExportConfig,
                 tensorboard_writer: SummaryWriter = None, metric_makers=None):

        self.export_config = export_config

        # Copies of things the trainer was given access to
        self.instance_problem = instance_problem

        # Helper objects
        self.tensorboard_writer = tensorboard_writer

        # Log directory / log files
        self.out_dir = out_dir
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not osp.exists(osp.join(self.out_dir, 'log.csv')):
            with open(osp.join(self.out_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        # Save instance problem
        self.instance_problem.save(self.instance_problem_path)

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

        self.val_losses_stored = []
        self.train_losses_stored = []
        self.joint_train_val_loss_mpl_figure = None  # figure for plotting losses on same plot
        self.iterations_for_losses_stored = []

        self.metric_makers = metric_makers

        # Writing activations

        self.run_loss_updates = True

        model_checkpoint_dir = osp.join(self.out_dir, 'model_checkpoints')
        os.mkdir(model_checkpoint_dir)
        self.model_history_saver = ModelHistorySaver(model_checkpoint_dir=model_checkpoint_dir,
                                                     interval_validate=self.export_config.interval_validate,
                                                     max_n_saved_models=self.export_config.max_n_saved_models)
        self.conservative_export_decider = ConservativeExportDecider(base_interval=self.export_config.interval_validate)

    @staticmethod
    def export_inst_sem_lbls_as_id2rgb(sem_lbls_as_batch_nparray, inst_lbls_as_batch_nparray, output_directory,
                                       image_names):
        return export_inst_sem_lbls_as_id2rgb(sem_lbls_as_batch_nparray, inst_lbls_as_batch_nparray, output_directory,
                                              image_names)

    def get_big_channel_set_to_fit_pred_and_gt(self, max_n_channels=(256 * 256), sem_vals_not_model_ids=True):
        """
        sem_vals_not_model_ids: False if you want what the model calls semantic ids (e.g. - remapping car to 3
        instead of 14); True if you want the sem vals from the initial dataset
        """

        n_stuff_classes = len(self.instance_problem.stuff_class_vals)
        n_thing_classes = len(self.instance_problem.thing_class_vals)
        n_thing_channels_per_cls = int((max_n_channels - n_stuff_classes) / n_thing_classes)
        channel_sem_vals, channel_inst_vals = [], []
        no_ch_0 = int(self.instance_problem.include_instance_channel0)  # 0 if channel 0, else 1
        for i, l in enumerate(self.instance_problem.labels_table):
            sem_val = l['id'] if sem_vals_not_model_ids else i
            is_thing = l['isthing']
            n_channels_this_cls = n_thing_channels_per_cls if is_thing else 1
            channel_sem_vals.extend([sem_val] * n_channels_this_cls)
            channel_inst_vals.extend([0] if not is_thing else list(range(no_ch_0, n_channels_this_cls + no_ch_0)))
        return channel_sem_vals, channel_inst_vals

    @property
    def instance_problem_path(self):
        return osp.join(self.out_dir, 'instance_problem_config.yaml')

    def write_eval_metrics(self, eval_metrics, loss, split, epoch, iteration):
        with open(osp.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [epoch, iteration] + [''] * 5 + \
                      [loss] + list(eval_metrics) + [elapsed_time]
            elif split == 'train':
                try:
                    eval_metrics_as_list = eval_metrics.tolist()
                except:
                    eval_metrics_as_list = list(eval_metrics)
                log = [epoch, iteration] + [loss] + eval_metrics_as_list + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def update_mpl_joint_train_val_loss_figure(self, train_loss, val_loss, iteration):
        assert train_loss is not None, ValueError
        assert val_loss is not None, ValueError
        figure_name = 'train/val losses'
        ylim_buffer_size = 3
        self.train_losses_stored.append(train_loss)
        self.val_losses_stored.append(val_loss)

        self.iterations_for_losses_stored.append(iteration)
        if self.joint_train_val_loss_mpl_figure is None:
            self.joint_train_val_loss_mpl_figure = plt.figure(figure_name)

        h = plt.figure(figure_name)

        plt.clf()
        train_label = 'train losses'  # TODO(allie): record number of images somewhere.. (we deleted it from here)
        val_label = 'val losses'

        plt.plot(self.iterations_for_losses_stored, self.train_losses_stored, label=train_label,
                 color=display_pyutils.GOOD_COLORS_BY_NAME['blue'])
        plt.plot(self.iterations_for_losses_stored, self.val_losses_stored, label=val_label,
                 color=display_pyutils.GOOD_COLORS_BY_NAME['aqua'])
        plt.xlabel('iteration')
        plt.legend()
        # Set y limits for just the last 10 datapoints
        last_x = max(len(self.train_losses_stored), len(self.val_losses_stored))
        if last_x >= 0:
            ymin = min(min(self.train_losses_stored[(last_x - ylim_buffer_size - 1):]),
                       min(self.val_losses_stored[(last_x - ylim_buffer_size - 1):]))
            ymax = max(max(self.train_losses_stored[(last_x - ylim_buffer_size - 1):]),
                       max(self.val_losses_stored[(last_x - ylim_buffer_size - 1):]))
        else:
            ymin, ymax = None, None
        if self.tensorboard_writer is not None:
            instanceseg.utils.export.log_plots(self.tensorboard_writer, 'joint_loss', [h], iteration)
        filename = os.path.join(self.out_dir, 'val_train_loss.png')
        h.savefig(filename)

        # zoom
        zoom_filename = os.path.join(self.out_dir, 'val_train_loss_zoom_last_{}.png'.format(ylim_buffer_size))
        if ymin is not None:
            plt.ylim(bottom=ymin, top=ymax)  # ymin, ymax, xmin, xmax deprecated Matplotlib 3.0
            plt.xlim(left=(last_x - ylim_buffer_size - 1), right=last_x)
            if self.tensorboard_writer is not None:
                instanceseg.utils.export.log_plots(self.tensorboard_writer,
                                                   'joint_loss_last_{}'.format(ylim_buffer_size),
                                                   [h], iteration)
            h.savefig(zoom_filename)
        else:
            shutil.copyfile(filename, zoom_filename)

    def retrieve_and_write_batch_activations(self, batch_input, iteration,
                                             get_activations_fcn):
        """
        get_activations_fcn: example in FCN8sInstance.get_activations(batch_input, layer_names)
        """
        if self.tensorboard_writer is not None:
            activations = get_activations_fcn(batch_input, self.export_config.activation_layers_to_export)
            histogram_activations = activations
            for name, activations in tqdm.tqdm(histogram_activations.items(),
                                               total=len(histogram_activations.items()),
                                               desc='Writing activation distributions', leave=False):
                if name == 'upscore8':
                    channel_labels = self.instance_problem.get_model_channel_labels('{}_{}')
                    assert activations.size(1) == len(channel_labels), '{} != {}'.format(activations.size(1),
                                                                                         len(channel_labels))
                    for c, channel_label in enumerate(channel_labels):
                        self.tensorboard_writer.add_histogram('batch_activations/{}/{}'.format(name, channel_label),
                                                              activations[:, c, :, :].cpu().numpy(),
                                                              iteration, bins='auto')
                elif name == 'conv1x1_instance_to_semantic':
                    channel_labels = self.instance_problem.get_channel_labels('{}_{}')
                    assert activations.size(1) == len(channel_labels)
                    for c, channel_label in enumerate(channel_labels):
                        try:
                            self.tensorboard_writer.add_histogram('batch_activations/{}/{}'.format(name, channel_label),
                                                                  activations[:, c, :, :].cpu().numpy(),
                                                                  iteration, bins='auto')
                        except IndexError as ie:
                            print('WARNING: Didn\'t write activations.  IndexError: {}'.format(ie))
                elif name == 'conv1_1':
                    # This is expensive to write, so we'll just write a representative set.
                    min = torch.min(activations)
                    max = torch.max(activations)
                    mean = torch.mean(activations)
                    representative_set = np.ndarray((100, 3))
                    representative_set[:, 0] = min
                    representative_set[:, 1] = max
                    representative_set[:, 2] = mean
                    self.tensorboard_writer.add_histogram('batch_activations/{}/min_mean_max_all_channels'.format(name),
                                                          representative_set, iteration, bins='auto')
                    continue

                self.tensorboard_writer.add_histogram('batch_activations/{}/all_channels'.format(name),
                                                      activations.cpu().numpy(), iteration, bins='auto')

    def write_loss_updates(self, old_loss, new_loss, old_assignments: LossMatchAssignments,
                           new_assignments: LossMatchAssignments, iteration):
        loss_improvement = old_loss - new_loss
        num_reassignments = float(
            (new_assignments.assigned_gt_inst_vals != old_assignments.assigned_gt_inst_vals).sum())
        self.tensorboard_writer.add_scalar('A_eval_metrics/train_minibatch_loss_improvement', loss_improvement,
                                           iteration)
        self.tensorboard_writer.add_scalar('A_eval_metrics/reassignment', num_reassignments, iteration)

    def compute_and_write_instance_metrics(self, model, iteration):
        if self.tensorboard_writer is not None:
            for split, metric_maker in tqdm.tqdm(self.metric_makers.items(), desc='Computing instance metrics',
                                                 total=len(self.metric_makers.items()), leave=False):
                metric_maker.clear()
                metric_maker.compute_metrics(model)
                metrics_as_nested_dict = metric_maker.get_aggregated_scalar_metrics_as_nested_dict()
                metrics_as_flattened_dict = flatten_dict(metrics_as_nested_dict)
                for name, metric in metrics_as_flattened_dict.items():
                    self.tensorboard_writer.add_scalar('C_{}_{}'.format(name, split), metric,
                                                       iteration)
                histogram_metrics_as_nested_dict = metric_maker.get_aggregated_histogram_metrics_as_nested_dict()
                histogram_metrics_as_flattened_dict = flatten_dict(histogram_metrics_as_nested_dict)
                if iteration != 0:  # screws up the axes if we do it on the first iteration with weird inits
                    for name, metric in tqdm.tqdm(histogram_metrics_as_flattened_dict.items(),
                                                  total=len(histogram_metrics_as_flattened_dict.items()),
                                                  desc='Writing histogram metrics', leave=False):
                        if torch.is_tensor(metric):
                            self.tensorboard_writer.add_histogram('C_instance_metrics_{}/{}'.format(split, name),
                                                                  metric.numpy(), iteration, bins='auto')
                        elif isinstance(metric, np.ndarray):
                            self.tensorboard_writer.add_histogram('C_instance_metrics_{}/{}'.format(split, name),
                                                                  metric, iteration, bins='auto')
                        elif metric is None:
                            import ipdb;
                            ipdb.set_trace()
                            pass
                        else:
                            raise ValueError('I\'m not sure how to write {} to tensorboard_writer (name is '
                                             '{}'.format(type(metric), name))

    def save_checkpoint(self, epoch, iteration, model, optimizer, best_mean_iu, mean_iu, out_dir=None,
                        save_by_iteration=True):
        out_name = 'checkpoint.pth.tar'
        out_dir = out_dir or os.path.join(self.out_dir)
        checkpoint_file = osp.join(out_dir, out_name)
        if hasattr(self, 'module'):
            model_state_dict = model.module.state_dict()  # nn.DataParallel
        else:
            model_state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'arch': model.__class__.__name__,
            'optim_state_dict': optimizer.state_dict(),
            'model_state_dict': model_state_dict,
            'best_mean_iu': best_mean_iu,
            'mean_iu': mean_iu
        }, checkpoint_file)

        self.model_history_saver.save_model_to_history(iteration, checkpoint_file)

        return checkpoint_file

    def copy_checkpoint_as_best(self, current_checkpoint_file, out_dir=None, out_name='model_best.pth.tar'):
        out_dir = out_dir or self.out_dir
        best_checkpoint_file = osp.join(out_dir, out_name)
        shutil.copy(current_checkpoint_file, best_checkpoint_file)
        return best_checkpoint_file

    def visualize_one_img_prediction_score(self, img_untransformed, softmax_scores, gt_sem_inst_tuple,
                                           channel_sem_values, channel_inst_vals, unassigned_gt_sem_inst_tuples):
        score_viz = visualization_utils.visualize_heatmaps(
            softmax_scores, gt_sem_inst_tuple, channel_sem_values, channel_inst_vals,
            sem_val_to_name=self.instance_problem.semantic_class_names_by_model_id,
            leftover_gt_sem_inst_tuples=unassigned_gt_sem_inst_tuples,
            input_image=img_untransformed,
            margin_color=(255, 255, 255), margin_size_small=3, margin_size_large=6,
            use_funky_void_pixels=True, void_val=-1)

        if self.export_config.downsample_multiplier_score_images != 1:
            score_viz = instanceseg.utils.imgutils.resize_img_by_multiplier(
                score_viz, self.export_config.downsample_multiplier_score_images)
        return score_viz

    def export_visualizations(self, visualizations, iteration, basename='val_', tile=True, out_dir=None):
        if visualizations is None:
            return
        out_dir = out_dir or osp.join(self.out_dir, 'visualization_viz')
        visualization_utils.export_visualizations(visualizations, out_dir, self.tensorboard_writer,
                                                  iteration, basename=basename, tile=tile)

    def run_post_train_iteration(self, full_input, loss_result: MatchingLossResult,
                                 epoch, iteration, new_loss_result: MatchingLossResult = None,
                                 get_activations_fcn=None, lrs_by_group=None, semantic_names_by_val=None):
        """
        get_activations_fcn=self.model.get_activations
        """
        eval_metrics = []
        if self.tensorboard_writer is not None:
            # TODO(allie): Check dimensionality of loss to prevent potential bugs
            self.tensorboard_writer.add_scalar('A_eval_metrics/train_minibatch_loss',
                                               loss_result.avg_loss.detach().sum(), iteration)

        if self.export_config.write_lr:
            for group_idx, lr in enumerate(lrs_by_group):
                self.tensorboard_writer.add_scalar('Z_hyperparameters/lr_group{}'.format(group_idx), lr, iteration)

        if self.export_config.export_component_losses:
            for c_idx, c_lbl in enumerate(self.instance_problem.get_model_channel_labels('{}_{}')):
                self.tensorboard_writer.add_scalar('B_channel_component_losses/train/{}'.format(c_lbl),
                                                   loss_result.loss_components_by_channel.detach()[:, c_idx].sum(),
                                                   iteration)
            for s_idx, s_val in enumerate(loss_result.semantic_vals):
                s_lbl = '({})_{}'.format(s_val, semantic_names_by_val[s_val]) if semantic_names_by_val is not None \
                    else '{}'.format(s_val)
                self.tensorboard_writer.add_scalar('B_semantic_component_losses/train/{}'.format(s_lbl),
                                                   loss_result.loss_components_by_sem_cls.detach()[:, s_idx].sum(),
                                                   iteration)

        if self.export_config.run_loss_updates:
            self.write_loss_updates(old_loss=loss_result.avg_loss.item(), new_loss=new_loss_result.avg_loss.item(),
                                    old_assignments=loss_result.assignments,
                                    new_assignments=new_loss_result.assignments, iteration=iteration)

            if self.export_config.export_activations and \
                    self.export_config.write_activation_condition(iteration, epoch):
                self.retrieve_and_write_batch_activations(batch_input=full_input, iteration=iteration,
                                                          get_activations_fcn=get_activations_fcn)
        return eval_metrics

    def run_post_val_iteration(self, imgs, sem_lbl, inst_lbl, score, assignments: LossMatchAssignments,
                               should_visualize, data_to_img_transformer):
        """
        data_to_img_transformer: img_untransformed, lbl_untransformed = f(img, lbl) : e.g. - resizes, etc.
        """
        true_labels, pred_labels, segmentation_visualizations, score_visualizations = [], [], [], []

        softmax_scores = F.softmax(score, dim=1).detach().cpu().numpy()
        pred_channel_lbl = score.detach().max(dim=1)[1].cpu().numpy()[:, :, :]
        lbl_true_sem, lbl_true_inst = (sem_lbl.detach().cpu(), inst_lbl.detach().cpu())
        if DEBUG_ASSERTS:
            assert pred_channel_lbl.shape == lbl_true_inst.shape
        max_n_insts_per_thing = 255
        for idx, (img, sem_lbl, inst_lbl, pred_l) in enumerate(zip(imgs, lbl_true_sem, lbl_true_inst,
                                                                   pred_channel_lbl)):
            # runtime_transformation needs to still run the resize, even for untransformed img, lbl pair
            img_untransformed, lbl_untransformed = data_to_img_transformer(img, (sem_lbl, inst_lbl)) \
                if data_to_img_transformer is not None else (img, (sem_lbl, inst_lbl))
            sem_lbl_np, inst_lbl_np = lbl_untransformed
            assert max_n_insts_per_thing >= inst_lbl.max()
            if should_visualize:
                segmentation_viz = visualization_utils.visualize_segmentations_as_rgb_imgs(
                    gt_sem_inst_lbl_tuple=(sem_lbl_np, inst_lbl_np),
                    pred_channelwise_lbl=pred_l,
                    channel_inst_vals=assignments.assigned_gt_inst_vals[idx, ...].numpy(),
                    channel_sem_vals=assignments.sem_values[idx, ...].numpy(),
                    unmatched_val=GT_VALUE_FOR_FALSE_POSITIVE,
                    instance_count_id_list=self.instance_problem.instance_count_id_list,
                    img=img_untransformed, overlay=False,
                    void_val=self.instance_problem.void_value)
                score_viz = self.visualize_one_img_prediction_score(
                    img_untransformed=img_untransformed, softmax_scores=softmax_scores[idx, ...],
                    gt_sem_inst_tuple=(sem_lbl_np, inst_lbl_np),
                    channel_sem_values=assignments.sem_values[idx, ...],
                    channel_inst_vals=assignments.assigned_gt_inst_vals[idx, ...],
                    unassigned_gt_sem_inst_tuples=assignments.unassigned_gt_sem_inst_tuples[idx])
                score_visualizations.append(score_viz)
                segmentation_visualizations.append(segmentation_viz)
        return segmentation_visualizations, score_visualizations

    def compute_eval_metrics(self, label_trues, label_preds):
        eval_metrics_list = instanceseg.utils.misc.label_accuracy_score(label_trues, label_preds,
                                                                        n_class=self.instance_problem.n_classes)
        return eval_metrics_list

    @staticmethod
    def untransform_data(data_loader, img, lbl):
        if data_loader.dataset.runtime_transformation is not None:
            runtime_transformation_undo = runtime_transformations.GenericSequenceRuntimeDatasetTransformer(
                [t for t in (data_loader.dataset.runtime_transformation.transformer_sequence or [])
                 if isinstance(t, runtime_transformations.BasicRuntimeDatasetTransformer)])
            img_untransformed, lbl_untransformed = runtime_transformation_undo.untransform(img, lbl)
        else:
            img_untransformed, lbl_untransformed = img, lbl
        return img_untransformed, lbl_untransformed

    def export_channelvals2d_as_id2rgb(self, labels_as_batch_nparray, output_directory, image_names):
        batch_sz = labels_as_batch_nparray.shape[0]
        for img_idx in range(batch_sz):
            lbl = labels_as_batch_nparray[img_idx, ...]
            sem_l, inst_l = self.instance_problem.decompose_semantic_and_instance_labels_with_original_sem_ids(lbl)
            img = sem_inst_lbls_to_id2rgb(sem_l, inst_l)
            out_file = os.path.join(output_directory, image_names[img_idx])
            visualization_utils.write_image(out_file, img)

    def export_rgb_images(self, images_as_batch_nparray, output_directory, image_names):
        batch_sz = images_as_batch_nparray.shape[0]
        for img_idx in range(batch_sz):
            img = images_as_batch_nparray[img_idx, ...]
            out_file = os.path.join(output_directory, image_names[img_idx])
            visualization_utils.write_image(out_file, img)

    def write_rgb_image(self, out_file, img):
        visualization_utils.write_image(out_file, img)

    @staticmethod
    def load_rgb_predictions_or_gt_to_id(in_file):
        rgb_img = imgutils.load_img_as_dtype(in_file, dtype=None)
        lbl = rgb2id(rgb_img)
        # convert void class
        lbl[255 + 256 * 255 + 256 * 256 * 255] = -1
        return

        # if self.exporter.export_config.export_activations:
        #     try:
        #         iteration = self.state.iteration
        #         upscore8_grad = self.model.upscore8.weight.grad
        #         for channel_idx, channel_name in enumerate(self.instance_problem.get_channel_labels()):
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'Z_upscore8_gradients/{}'.format(channel_idx),
        #                 upscore8_grad[channel_idx, :, :, :], self.state.iteration)
        #         score_pool4_weight_grad = self.model.score_pool4.weight.grad
        #         score_pool4_bias_grad = self.model.score_pool4.bias.grad
        #         assert len(score_pool4_bias_grad) == self.instance_problem.n_classes
        #         for channel_idx, channel_name in enumerate(self.instance_problem.get_channel_labels()):
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'Z_score_pool4_weight_gradients/{}'.format(channel_idx),
        #                 score_pool4_weight_grad[channel_idx, :, :, :], self.state.iteration)
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'score_pool4_bias_gradients/{}'.format(channel_idx),
        #                 score_pool4_bias_grad[channel_idx], self.state.iteration)
        #     except:
        #         import ipdb;
        #         ipdb.set_trace()
        #         raise
