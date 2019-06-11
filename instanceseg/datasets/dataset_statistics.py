import numpy as np
import torch
import tqdm
import cv2
import abc
import logging
import os, shutil
from instanceseg.analysis import visualization_utils

try:
    from tabulate import tabulate
except:
    tabulate = None

logger = logging.getLogger(__name__)


class DatasetStatisticCacheInterface(object):
    """
    Stores some statistic about a set of images contained in a Dataset class (e.g. - number of instances in the image).
    Inheriting from this handles some of the save/load/caching.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cache_file=None, override=False):
        self._stat_tensor = None
        self.cache_file = cache_file
        self.override = override

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.stat_tensor.shape

    @property
    def n_images(self):
        return self.shape[0]

    @staticmethod
    def load(stats_filename):
        return torch.from_numpy(np.load(stats_filename))

    @staticmethod
    def save(statistics, stats_filename):
        np.save(stats_filename, statistics)

    @property
    def stat_tensor(self):
        if self._stat_tensor is None:
            raise Exception(
                'Statistic has not yet been computed.  Run {}.compute(<dataset>)'.format(
                    self.__class__.__name__))
        return self._stat_tensor

    def compute_or_retrieve(self, dataset):
        if self.cache_file is None:
            logger.info('Computing statistics without cache')
            self._stat_tensor = self._compute(dataset)
        elif self.override or not os.path.exists(self.cache_file):
            logger.info('Computing statistics for file {}'.format(self.cache_file))
            self._stat_tensor = self._compute(dataset)
            self.save(self._stat_tensor, self.cache_file)
        else:
            logger.info('Loading statistics from file {}'.format(self.cache_file))
            self._stat_tensor = self.load(self.cache_file)

    @abc.abstractmethod
    def _compute(self, dataset):
        raise NotImplementedError

    def print_stat_tensor_for_txt_storage(self, stat_tensor):
        print(stat_tensor)

    def pprint_stat_tensor(self, stat_tensor=None, labels=None, with_labels=True):
        if stat_tensor is None:
            stat_tensor = self.stat_tensor
        n_columns = stat_tensor.size(1)
        if with_labels:
            try:
                headings = labels if labels is not None else self.labels
            except NotImplementedError:
                headings = ['{}'.format(x) for x in range(stat_tensor.size(1))]
            assert n_columns == len(headings)
            nested_list = [headings]
        else:
            nested_list = []
        nested_list += stat_tensor.tolist()
        if tabulate is None:
            print(Warning('pretty print only works with tabulate installed'))
            for l in nested_list:
                print('\t'.join(['{}'.format(x) for x in l]))
        else:
            print(tabulate(nested_list))


class PixelsPerSemanticClass(DatasetStatisticCacheInterface):

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False):
        super(PixelsPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals

    @property
    def labels(self):
        return self.semantic_class_names

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        semantic_pixel_counts = self.compute_semantic_pixel_counts(dataset,
                                                                   self.semantic_class_vals)
        self._stat_tensor = semantic_pixel_counts
        tensor_size = torch.Size((len(dataset), len(self.semantic_class_vals)))
        assert semantic_pixel_counts.size() == tensor_size, \
            'semantic pixel counts should be a matrix of size {}, not {}'.format(
                tensor_size, semantic_pixel_counts.size())
        return semantic_pixel_counts

    @staticmethod
    def compute_semantic_pixel_counts(dataset, semantic_class_vals):
        semantic_pixel_counts_nested_list = []
        for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(
                enumerate(dataset), total=len(dataset),
                desc='Running semantic pixel statistics on dataset'.format(dataset), leave=False):
            semantic_pixel_counts_nested_list.append([(sem_lbl == sem_val).sum() for sem_val in \
                                                      semantic_class_vals])
        semantic_pixel_counts = torch.IntTensor(semantic_pixel_counts_nested_list)
        return semantic_pixel_counts


class NumberofInstancesPerSemanticClass(DatasetStatisticCacheInterface):
    """
    Computes NxS nparray: For each of N images, contains the number of instances of each of S
    semantic classes
    """

    def __init__(self, semantic_classes, cache_file=None, override=False):
        super(NumberofInstancesPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_classes = semantic_classes

    @property
    def labels(self):
        return self.semantic_classes

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        instance_counts = self.compute_instance_counts(dataset, self.semantic_classes)
        self._stat_tensor = instance_counts
        return instance_counts

    @staticmethod
    def compute_instance_counts(dataset, semantic_classes):
        instance_counts = torch.ones(len(dataset), len(semantic_classes)) * -1
        for idx, (img, (sem_lbl, inst_lbl)) in \
                tqdm.tqdm(enumerate(dataset), total=len(dataset),
                          desc='Running instance statistics on dataset'.format(dataset),
                          leave=False):
            for sem_idx, sem_val in enumerate(semantic_classes):
                sem_locations_bool = sem_lbl == sem_val
                if torch.sum(sem_locations_bool) > 0:
                    my_max = inst_lbl[sem_locations_bool].max()
                    instance_counts[idx, sem_idx] = my_max
                else:
                    instance_counts[idx, sem_idx] = 0
                if sem_idx == 0 and instance_counts[idx, sem_idx] > 0:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception('inst_lbl should be 0 wherever sem_lbl is 0')
        return instance_counts


def get_occlusions_hws_from_labels(sem_lbl_np, inst_lbl_np, semantic_class_vals):
    """
    Returns (h,w,S) tensor where S is the number of semantic classes
    """
    # Make into batch form to use more general functions
    sem_lbl_np = sem_lbl_np[:, :, None]
    inst_lbl_np = inst_lbl_np[:, :, None]
    list_of_occlusion_locations_per_cls = []
    n_occlusion_pairings_per_sem_cls = np.zeros(len(semantic_class_vals))
    for sem_idx, sem_val in enumerate(semantic_class_vals):
        n_occlusion_pairings, all_occlusion_locations = \
            OcclusionsOfSameClass.compute_occlusions_from_batch_of_one_semantic_cls(
                sem_lbl_np, inst_lbl_np, sem_val)
        assert all_occlusion_locations.shape[2] == 1
        list_of_occlusion_locations_per_cls.append(all_occlusion_locations[:, :, 0])
        n_occlusion_pairings_per_sem_cls[sem_idx] = n_occlusion_pairings
    return n_occlusion_pairings_per_sem_cls, np.stack(list_of_occlusion_locations_per_cls, axis=2)


class OcclusionsOfSameClass(DatasetStatisticCacheInterface):
    default_compute_batch_sz = 10
    debug_dir = None

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False, compute_batch_size=None, debug=False):
        super(OcclusionsOfSameClass, self).__init__(cache_file, override)
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals
        self.compute_batch_size = compute_batch_size or self.default_compute_batch_sz
        self.debug = debug

    @property
    def labels(self):
        return [s.replace(' ', '_') for s in self.semantic_class_names]

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        occlusion_counts = self.compute_occlusion_counts(dataset, self.semantic_class_vals,
                                                         self.compute_batch_size)
        self._stat_tensor = occlusion_counts
        assert occlusion_counts.size() == torch.Size((len(dataset), len(self.semantic_class_vals)))
        return occlusion_counts

    @staticmethod
    def dilate(img_hwc, kernel=np.ones((3, 3)), iterations=1, dst=None):
        if dst is None:
            dilated_img = cv2.dilate(img_hwc, kernel=kernel, iterations=iterations)
            if img_hwc.shape[2] == 1:
                dilated_img = np.expand_dims(dilated_img, axis=2)
        else:
            cv2.dilate(img_hwc, kernel=kernel, iterations=iterations, dst=dst)
            if img_hwc.shape[2] == 1:
                np.expand_dims(dst, axis=2)
            dilated_img = None
        return dilated_img

    def compute_occlusion_counts(self, dataset, semantic_class_vals=None, compute_batch_size=None,
                                 debug=None):
        debug = debug if debug is not None else self.debug
        semantic_classes = semantic_class_vals or range(dataset.n_semantic_classes)
        # noinspection PyTypeChecker
        occlusion_counts = torch.zeros((
            len(dataset), len(semantic_class_vals)), dtype=torch.int)
        batch_size = compute_batch_size or self.default_compute_batch_sz
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=None, num_workers=4)
        batch_img_idx = 0
        for batch_idx, (_, (sem_lbl_batch, inst_lbl_batch)) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='Running occlusion statistics on dataset'.format(dataset), leave=False):
            batch_sz = sem_lbl_batch.shape[0]
            # Populates occlusion_counts
            batch_occlusion_counts = self.compute_occlusions_from_batch(
                sem_lbl_batch, inst_lbl_batch, semantic_classes, start_img_idx=batch_img_idx,
                debug=debug)
            occlusion_counts[batch_img_idx:(batch_img_idx + batch_sz), :] = torch.from_numpy(
                batch_occlusion_counts)
            batch_img_idx += batch_sz
        return occlusion_counts

    @staticmethod
    def torch_label_batch_to_np_batch_for_dilation(tensor):
        return tensor.numpy().astype(np.uint8).transpose(1, 2, 0)

    def compute_occlusions_from_batch(self, sem_lbl_batch, inst_lbl_batch, semantic_classes,
                                      start_img_idx=None, debug=False):
        batch_sz = sem_lbl_batch.size(0)
        batch_occlusion_counts = np.zeros((batch_sz, len(semantic_classes)), dtype=int)
        # h x w x b (for dilation)
        inst_lbl_np = self.torch_label_batch_to_np_batch_for_dilation(inst_lbl_batch)
        sem_lbl_np = self.torch_label_batch_to_np_batch_for_dilation(sem_lbl_batch)

        for sem_idx in semantic_classes:
            n_occlusion_pairings, occlusion_locs = \
                self.compute_occlusions_from_batch_of_one_semantic_cls(sem_lbl_np,
                                                                       inst_lbl_np, sem_idx)
            batch_occlusion_counts[:, sem_idx] = n_occlusion_pairings
            if debug:
                occlusion_and_sem_cls = self.dilate(occlusion_locs.astype('uint8'),
                                                    iterations=4) + \
                                        (sem_lbl_np == sem_idx).astype('uint8')
                self.export_debug_images_from_batch(
                    occlusion_and_sem_cls, ['occlusion_locations_{}_{}_n_occlusions_{}'.format(
                        start_img_idx + i, self.semantic_class_names[sem_idx],
                        n_occlusion_pairings[i])
                        for i in range(batch_sz)])

        # dilate occlusion locations for visibility
        return batch_occlusion_counts

    @classmethod
    def compute_occlusions_from_batch_of_one_semantic_cls(cls, sem_lbl_np, inst_lbl_np, sem_idx):
        batch_sz = sem_lbl_np.shape[2]
        all_occlusion_locations = np.zeros_like(sem_lbl_np).astype(int)
        n_occlusion_pairings = np.zeros(batch_sz, dtype=int)
        # sem_lbl_np, inst_lbl_np: h x w x b
        if cls.cannot_have_occlusions(sem_lbl_np, inst_lbl_np, sem_idx):
            return n_occlusion_pairings, all_occlusion_locations  # 0, zeros
        semantic_cls_bool = sem_lbl_np == sem_idx
        max_num_instances = inst_lbl_np[semantic_cls_bool].max()
        dilated_instance_masks = []
        possible_instance_values = range(1, max_num_instances + 1)

        # Collect all the instance masks
        for inst_val in possible_instance_values:
            # noinspection PyTypeChecker
            inst_loc_bool = cls.intersect_two_binary_masks(semantic_cls_bool,
                                                           (inst_lbl_np == inst_val))
            if not np.any(inst_loc_bool):
                continue
            else:
                dilated_instance_masks.append(cls.dilate(inst_loc_bool.astype(np.uint8),
                                                         kernel=np.ones((3, 3)), iterations=1))
        # Compute pairwise occlusions
        for dilate_idx1 in range(len(dilated_instance_masks)):
            for dilate_idx2 in range(dilate_idx1 + 1, len(dilated_instance_masks)):
                mask_pair_intersection = cls.intersect_two_binary_masks(
                    dilated_instance_masks[dilate_idx1], dilated_instance_masks[dilate_idx2])
                all_occlusion_locations += mask_pair_intersection
                #  NOTE(allie): Below is the computationally expensive line.
                n_occlusion_pairings += (np.any(mask_pair_intersection, axis=(0, 1))).astype(int)

        # n_occlusion_pairings: (b,) ,  all_occlusion_locations: (h,w,b)
        return n_occlusion_pairings, all_occlusion_locations

    @staticmethod
    def clear_and_create_dir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

    def prep_debug_dir(self):
        if self.debug_dir is None:
            self.debug_dir = '/tmp/occlusion_debug/'
            self.clear_and_create_dir(self.debug_dir)

    @staticmethod
    def export_label_image(img, filename):
        visualization_utils.write_label(filename, img)

    def export_debug_image(self, img, basename):
        filename = os.path.join(self.debug_dir, '{}.png'.format(basename))
        self.export_label_image(img, filename)

    def export_debug_images_from_batch(self, imgs_as_batch, basenames):
        self.prep_debug_dir()
        batch_sz = imgs_as_batch.shape[2]
        assert len(basenames) == batch_sz
        for img_idx in range(batch_sz):
            self.export_debug_image(imgs_as_batch[:, :, img_idx], basenames[img_idx])

    @staticmethod
    def cannot_have_occlusions(sem_lbl_batch, inst_lbl_batch, sem_idx):
        # Check if it even contains this semantic class
        if (sem_lbl_batch == sem_idx).sum() == 0:
            return True
        # Check if it contains at least two instances
        if inst_lbl_batch[sem_lbl_batch == sem_idx].max() < 2:
            return True
        return False

    @staticmethod
    def intersect_two_binary_masks(mask1: np.ndarray, mask2: np.ndarray):
        # would check that only int values are in (0,1), but too much computation.
        for mask in [mask1, mask2]:
            assert mask.dtype in [np.bool, np.int, np.uint8], \
                'I didnt expect a boolean mask with dtype {}'.format(mask1.dtype)
        return mask1 * mask2

        # if debug:
        #     for img_idx in range(batch_sz):
        #         visualization_utils.write_label(os.path.join(
        #             tmpdir, 'intermediate_union_before_img_{}_cls_{}.png'.format(
        #                 img_idx + compute_batch_size * batch_idx,
        #                 self.semantic_class_names[sem_idx])),
        #             intermediate_union[:, :, img_idx])

        # if debug:
        #     for img_idx in range(batch_sz):
        #         visualization_utils.write_label(os.path.join(
        #             tmpdir, 'dilated_instance_mask_{}_cls_{}_inst_{}.png'.format(
        #                 img_idx + compute_batch_size * batch_idx,
        #                 self.semantic_class_names[sem_idx], inst_val)),
        #             intersections_with_previous_instances[:, :, img_idx])
        #
        # if debug:
        #     for img_idx in range(batch_sz):
        #         visualization_utils.write_label(os.path.join(
        #             tmpdir, 'intermediate_union_after_img_{}_cls_{}.png'.format(
        #                 img_idx + compute_batch_size * batch_idx,
        #                 self.semantic_class_names[sem_idx])),
        #             intermediate_union[:, :, img_idx])
        #
