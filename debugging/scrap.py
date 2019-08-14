import glob

import numpy as np
import os
import torch
import yaml
import tqdm
from PIL import Image

from instanceseg.ext.panopticapi.utils import rgb2id
from instanceseg.losses import loss
from instanceseg.utils import instance_utils
from instanceseg.utils.misc import y_or_n_input
from scripts.convert_test_results_to_coco import get_outdirs_cache_root


def load_gt_img_in_panoptic_form(gt_img_file):
    with Image.open(gt_img_file) as img:
        pan_gt = np.array(img, dtype=np.uint32)
    pan_gt = rgb2id(pan_gt)
    return pan_gt


def main(test_outdir):
    scores_outdir = os.path.join(test_outdir, 'scores')
    groundtruth_outdir = os.path.join(test_outdir, 'groundtruth')

    test_cfg_file = os.path.join(test_outdir, 'config.yaml')
    problem_config_file = os.path.join(test_outdir, 'instance_problem_config.yaml')

    with open(os.path.join(test_outdir, 'train_logdir.txt'), 'r') as fid:
        train_logdir = fid.read().strip()

    analysis_cache_outdir = get_outdirs_cache_root(train_logdir=train_logdir, test_pred_outdir=scores_outdir)
    dataset_cache_dir = os.path.dirname(os.path.dirname(analysis_cache_outdir.rstrip('/')))
    if not os.path.exists(dataset_cache_dir):
        raise Exception('Dataset cache doesnt exist.  Maybe the dataset name is wrong? {}'.format(dataset_cache_dir))
    if not os.path.exists(analysis_cache_outdir):
        os.makedirs(analysis_cache_outdir)

    compiled_loss_arr_outfile = os.path.join(analysis_cache_outdir, 'losses')
    if os.path.exists(compiled_loss_arr_outfile):
        if not y_or_n_input('Already found {}.  Overwrite?(y/N)'.format(compiled_loss_arr_outfile), default='n',
                            convert_to_bool_is_y=True):
            return

    cfg = yaml.safe_load(open(test_cfg_file, 'rb'))
    problem_config = instance_utils.InstanceProblemConfig.load(problem_config_file)

    score_files = sorted(glob.glob(os.path.join(scores_outdir, '*.pt')))
    gt_files = sorted(glob.glob(os.path.join(groundtruth_outdir, '*.png')))
    assert len(score_files) == len(gt_files)

    my_loss_object = loss.loss_object_factory(
        loss_type=cfg['loss_type'], semantic_instance_class_list=problem_config.semantic_instance_class_list,
        instance_id_count_list=problem_config.instance_count_id_list, matching=cfg['matching'],
        size_average=cfg['size_average'])

    n_images = len(score_files)
    losses_compiled_per_img_per_cls = -1 * np.ones((n_images, problem_config.n_semantic_classes))
    losses_compiled_per_img_per_channel = -1 * np.ones((n_images, problem_config.n_classes))

    for idx, (score_file, gt_file) in tqdm.tqdm(enumerate(zip(score_files, gt_files)), total=n_images,
                                                desc='Getting losses for saved scores, GT'):
        score = torch.load(score_file)
        gt_im = load_gt_img_in_panoptic_form(gt_file).astype('int')
        gt_sem, gt_inst = problem_config.decompose_semantic_and_instance_labels(gt_im)
        score = score.view(1, *score.shape)
        gt_sem, gt_inst = torch.Tensor(gt_sem[np.newaxis, ...]).cuda(device=score.device), \
                          torch.Tensor(gt_inst[np.newaxis, ...]).cuda(device=score.device)
        pred_permutations, total_loss, loss_components = my_loss_object.loss_fcn(score, gt_sem, gt_inst)
        per_channel_loss = loss_components.cpu().numpy()
        per_cls_loss = problem_config.aggregate_across_same_sem_cls(per_channel_loss)
        losses_compiled_per_img_per_channel[idx, :] = per_channel_loss
        losses_compiled_per_img_per_cls[idx, :] = per_cls_loss

    d = {
        'losses_compiled_per_img_per_cls': losses_compiled_per_img_per_cls,
        'losses_compiled_per_img_per_channel': losses_compiled_per_img_per_channel,
        'problem_config': problem_config,
        'scores_outdir': scores_outdir,
        'groundtruth_outdir': groundtruth_outdir
    }

    np.savez(compiled_loss_arr_outfile, **d)
    print('Losses saved to {}'.format(compiled_loss_arr_outfile))
    return d


if __name__ == '__main__':
    test_outdir = 'scripts/logs/synthetic/test_2019-08-12-131034_VCS-c8923f1__test_split-val'
    d = main(test_outdir)
