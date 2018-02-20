import torch
import argparse
import os
from examples import script_utils
import torchfcn
import local_pyutils
import numpy as np
from torch.autograd import Variable
from torchfcn import losses
from torchfcn import visualization_utils

logger = local_pyutils.get_logger

here = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

default_configuration = dict(
    max_iteration=10000,
    lr=1.0e-5,
    weight_decay=5e-6,
    interval_validate=100,
    n_max_per_class=3,
    n_training_imgs=1000,
    n_validation_imgs=50,
    batch_size=1,
    recompute_optimal_loss=False,
    size_average=True,
    val_on_train=True,
    map_to_semantic=False,
    matching=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('--logdir', default=None, help='Log directory; must either contain ' \
                                                       'model_best.pth.tar or '
                                                       'checkpoint.pth.tar and config.yaml')
    args = parser.parse_args()
    logdir = args.logdir
    if logdir is None:
        log_parent_directory = os.path.join(here, 'logs')
        logdir = script_utils.get_latest_logdir_with_checkpoint(log_parent_directory)
        if logdir is None:
            raise Exception('No logs with checkpoint found in {}'.format(log_parent_directory))
    args.logdir = logdir
    return args


def configured_cross_entropy(semantic_instance_class_list, size_average=True, matching_loss=True,
                             recompute_loss_at_optimal_permutation=False, **kwargs):
    configured_function = lambda scores, target: losses.cross_entropy2d(
        scores, target,
        semantic_instance_labels=semantic_instance_class_list,
        matching=matching_loss, size_average=size_average,
        recompute_optimal_loss=recompute_loss_at_optimal_permutation, **kwargs)
    return configured_function


def predict_one_batch(model, data, target, loss_fcn, data_loader, n_class,
                      should_visualize=True, cuda=True):
    if cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    try:
        score = model(data)
    except:
        import ipdb; ipdb.set_trace()
        raise

    gt_permutations, loss = loss_fcn(score, target)
    if np.isnan(float(loss.data[0])):
        raise ValueError('loss is nan while validating')
    val_loss = float(loss.data[0]) / len(data)

    imgs = data.data.cpu()
    # numpy_score = score.data.numpy()
    lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]
    # confidence = numpy_score[numpy_score == numpy_score.max(dim=1)[0]]
    lbl_true = target.data.cpu()
    true_labels, pred_labels, visualizations = [], [], []

    for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
        img, lt = data_loader.dataset.untransform(img, lt)
        true_labels.append(lt)
        pred_labels.append(lp)
        if should_visualize:
            viz = visualization_utils.visualize_segmentation(
                lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                overlay=False)
            visualizations.append(viz)
    return val_loss, lbl_pred, lbl_true, visualizations


def main():
    num_train_images = 10
    num_val_images = 5
    args = parse_args()
    logdir = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    model_path = script_utils.get_latest_model_path_from_logdir(logdir)
    config_path = os.path.join(logdir, 'config.yaml')
    assert os.path.exists(config_path), '{} doesn\'t exist'.format(config_path)
    cfg = default_configuration
    cfg.update(script_utils.load_config(config_path))
    print(cfg)

    # Load datasets
    root = os.path.expanduser('~/data/cityscapes')
    dataset_kwargs = dict(n_max_per_class=cfg['n_max_per_class'], batch_size=cfg['batch_size'])
    train_loader_for_val = script_utils.get_cityscapes_train_loader(
        root, modified_length=num_train_images, **dataset_kwargs)
    val_loader = script_utils.get_cityscapes_val_loader(root, modified_length=num_val_images,
                                                        **dataset_kwargs)
    # TODO(allie): verify cityscapes class names come out correct here...
    n_semantic_classes_with_background = len(train_loader_for_val.dataset.class_names)

    # Create model
    model = torchfcn.models.FCN8sInstance(
        n_semantic_classes_with_background=n_semantic_classes_with_background,
        n_max_per_class=cfg['n_max_per_class'], map_to_semantic=cfg['map_to_semantic'])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    previous_epoch, previous_iteration = checkpoint['epoch'], checkpoint['iteration']

    output_dir = os.path.join(logdir, 'analysis')
    local_pyutils.mkdir_if_needed(output_dir)

    my_cross_entropy_loss = configured_cross_entropy(
        semantic_instance_class_list=model.semantic_instance_class_list)

    for batch_idx, (data, target) in enumerate(train_loader_for_val):
        val_loss, lbl_pred, lbl_true, visualizations = \
            predict_one_batch(model, data, target, my_cross_entropy_loss, train_loader_for_val,
                              n_class=model.n_classes, should_visualize=True, cuda=cuda)
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
