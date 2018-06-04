import os.path as osp
from torchfcn.datasets import dataset_utils
import numpy as np

here = osp.dirname(__file__)


def test_lr_ordering():
    img_id = '2011_003114'
    test_data_folder = osp.join(osp.dirname(here), 'test_data')
    sem_lbl_file = osp.join(test_data_folder, 'VOC2012', 'SegmentationClass', '{}.png'.format(img_id))
    inst_lbl_file_unordered = sem_lbl_file.replace('Class', 'Object').replace('.png', '_per_sem_cls.png')
    out_inst_lbl_file_ordered = '/tmp/{}_per_sem_cls_ordered_LR.png'.format(img_id)
    dataset_utils.generate_lr_ordered_instance_file(inst_lbl_file_unordered=inst_lbl_file_unordered,
                                                    sem_lbl_file=sem_lbl_file,
                                                    out_inst_lbl_file_ordered=out_inst_lbl_file_ordered)
    sem_lbl = dataset_utils.load_img_as_dtype(sem_lbl_file, np.int32)
    inst_lbl_ordered = dataset_utils.load_img_as_dtype(out_inst_lbl_file_ordered, np.int32)
    inst_lbl_unordered = dataset_utils.load_img_as_dtype(inst_lbl_file_unordered, np.int32)

    unique_sem_lbls = np.unique(sem_lbl)
    sem_cls_preordered = []
    for sem_val in unique_sem_lbls[unique_sem_lbls > 0]:
        unordered_unique_instance_idxs = sorted(np.unique(inst_lbl_unordered[sem_lbl == sem_val]))
        ordered_unique_instance_idxs = sorted(np.unique(inst_lbl_ordered[sem_lbl == sem_val]))

        assert not np.any(unordered_unique_instance_idxs == 0)
        unordered_coms = []
        for inst_val in unordered_unique_instance_idxs:
            com = dataset_utils.compute_centroid_binary_mask(np.logical_and(sem_lbl == sem_val,
                                                                            inst_lbl_unordered == inst_val))
            unordered_coms.append(com)
        unordered_left_right_ordering = [x for x in np.argsort([com[1] for com in unordered_coms])]
        ordered_coms = []
        for inst_val in ordered_unique_instance_idxs:
            com = dataset_utils.compute_centroid_binary_mask(np.logical_and(sem_lbl == sem_val,
                                                                            inst_lbl_ordered == inst_val))
            ordered_coms.append(com)
        ordered_left_right_ordering = [x for x in np.argsort([com[1] for com in ordered_coms])]

        # Assert that they're actually in order
        sem_cls_preordered.append(all([x == y for x, y in zip(unordered_left_right_ordering, list(range(len(
            unordered_left_right_ordering))))]))
        assert all([x == y for x, y in zip(ordered_left_right_ordering, list(range(len(
            ordered_left_right_ordering))))]), Exception('Left-right ordering failed')
        print('sem_cls={}:\n{} -> {}'.format(sem_val, unordered_left_right_ordering, ordered_left_right_ordering))
    if all(sem_cls_preordered):
        raise ValueError('Chose a bad image (already ordered).')


if __name__ == '__main__':
    test_lr_ordering()

