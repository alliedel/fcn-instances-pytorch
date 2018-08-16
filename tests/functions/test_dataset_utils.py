import os.path as osp
from instanceseg.utils import datasets
import numpy as np

here = osp.dirname(__file__)


def is_lr_ordered(sem_lbl, inst_lbl):
    unique_sem_lbls = np.unique(sem_lbl)
    sem_cls_preordered = []
    ordering_correct = True
    for sem_val in unique_sem_lbls[unique_sem_lbls > 0]:
        unique_instance_idxs = sorted(np.unique(inst_lbl[sem_lbl == sem_val]))
        assert 0 not in unique_instance_idxs

        ordered_coms = []
        for inst_val in unique_instance_idxs:
            com = datasets.compute_centroid_binary_mask(np.logical_and(sem_lbl == sem_val,
                                                                       inst_lbl == inst_val))
            ordered_coms.append(com)
        ordered_left_right_ordering = [x for x in np.argsort([com[1] for com in ordered_coms])]

        # Assert that they're actually in order
        sem_cls_preordered.append(all([x == y for x, y in zip(ordered_left_right_ordering, list(range(len(
            ordered_left_right_ordering))))]))
        if not all([x == y for x, y in zip(ordered_left_right_ordering, list(range(len(
                ordered_left_right_ordering))))]):
            ordering_correct = False
            break
    return ordering_correct


def test_lr_ordering_voc():
    img_id = '2011_003114'
    test_data_folder = osp.join(osp.dirname(here), 'test_data')
    sem_lbl_file = osp.join(test_data_folder, 'VOC2012', 'SegmentationClass', '{}.png'.format(img_id))
    inst_lbl_file_unordered = sem_lbl_file.replace('Class', 'Object').replace('.png', '_per_sem_cls.png')
    out_inst_lbl_file_ordered = '/tmp/{}_per_sem_cls_ordered_LR.png'.format(img_id)
    datasets.generate_lr_ordered_instance_file(inst_lbl_file_unordered=inst_lbl_file_unordered,
                                               sem_lbl_file=sem_lbl_file,
                                               out_inst_lbl_file_ordered=out_inst_lbl_file_ordered)
    sem_lbl = datasets.load_img_as_dtype(sem_lbl_file, np.int32)
    inst_lbl_ordered = datasets.load_img_as_dtype(out_inst_lbl_file_ordered, np.int32)
    inst_lbl_unordered = datasets.load_img_as_dtype(inst_lbl_file_unordered, np.int32)

    if is_lr_ordered(sem_lbl, inst_lbl_unordered):
        raise ValueError('Chose a bad image (already ordered).')

    if not is_lr_ordered(sem_lbl, inst_lbl_ordered):
        raise ValueError('Instance labels were incorrectly ordered.')
    print('PASSED')


if __name__ == '__main__':
    test_lr_ordering_voc()

