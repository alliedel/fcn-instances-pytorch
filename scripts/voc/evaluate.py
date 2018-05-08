#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
import tqdm
from torch.autograd import Variable

import torchfcn
from torchfcn import visualization_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-m', '--model', help='Model type (e.g. - fcn8s)', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/data/datasets')
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)
    model_dir = osp.basename(osp.dirname(model_file))
    if args.model is None:
        if model_dir.startswith('MODEL-fcn32s'):
            model = torchfcn.models.FCN32s(n_class=21)
        elif model_dir.startswith('MODEL-fcn16s'):
            model = torchfcn.models.FCN16s(n_class=21)
        elif model_dir.startswith('MODEL-fcn8s'):
            if osp.basename(osp.dirname(model_file)).startswith('MODEL-fcn8s-atonce'):
                model = torchfcn.models.FCN8sAtOnce(n_class=21)
            else:
                model = torchfcn.models.FCN8s(n_class=21)
        else:
            raise ValueError('unknown model for file {}'.format(osp.basename(model_dir)))
    else:
        if args.model == 'fcn32s':
            model = torchfcn.models.FCN32s(n_class=21)
        elif args.model == 'fcn16s':
            model = torchfcn.models.FCN16s(n_class=21)
        elif args.model == 'fcn8s-at-once':
            model = torchfcn.models.FCN8sAtOnce(n_class=21)
        elif args.model == 'fcn8s':
            model = torchfcn.models.FCN8s(n_class=21)
        else:
            raise ValueError('unknown model type {}'.format(args.model))
        
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = visualization_utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names,
                    overlay=val_loader.visualize_overlay)
                visualizations.append(viz)
    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = visualization_utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
