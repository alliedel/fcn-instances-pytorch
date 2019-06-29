from . import labels_table_cityscapes
import numpy as np
import os.path as osp
import os
import PIL.Image
from instanceseg.utils import misc, datasets
import six.moves


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def create_label_rgb_list(n_classes):
    cmap = np.zeros((n_classes, 3))
    for i in six.moves.range(0, n_classes):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

# def convert_to_p_mode_file(old_file, new_file, palette, assert_inside_palette_range=True):
#     im = PIL.Image.open(old_file)
#     if assert_inside_palette_range:
#         max_palette_val = int(len(palette.getpalette()) / 3 - 1)
#         min_val, max_val = im.getextrema()
#         assert min_val >= 0
#         assert max_val <= max_palette_val, '{}:\nmax value, {}, > palette max, {}'.format(old_file, max_val,
#                                                                                           max_palette_val)
#
#     if im.mode == 'P':  # already mode p.  symlink so we dont go through this again.
#         os.symlink(old_file, new_file)
#     elif im.mode == 'I':
#         arr = np.array(im)
#         datasets.write_np_array_as_img_with_colormap_palette(arr, new_file, palette)
#     else:  # if im.mode == 'RGB':
#         converted = im.quantize(palette=palette)
#         converted.save(new_file)
#
#
# class ConvertLblstoPModePILImages(object):
#     old_sem_file_tag = '.png'
#     new_sem_file_tag = '_mode_p.png'
#     old_inst_file_tag = '.png'
#     new_inst_file_tag = '_mode_p.png'
#
#     def __init__(self, semantic_palette, instance_palette):
#         self.semantic_palette = labels_table_cityscapes.get_semantic_palette_image()
#         self.instance_palette = labels_table_cityscapes.get_instance_palette_image()
#
#     def transform(self, img_file, sem_lbl_file, inst_lbl_file):
#         assert self.old_sem_file_tag in sem_lbl_file and self.old_sem_file_tag in inst_lbl_file
#         new_sem_lbl_file = sem_lbl_file.replace(self.old_sem_file_tag, self.new_sem_file_tag)
#         if not osp.isfile(new_sem_lbl_file):
#             assert osp.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
#             print('Creating {} from {}'.format(new_sem_lbl_file, sem_lbl_file))
#             convert_to_p_mode_file(sem_lbl_file, new_sem_lbl_file, palette=self.semantic_palette,
#                                    assert_inside_palette_range=True)
#         new_inst_lbl_file = inst_lbl_file.replace(self.old_inst_file_tag, self.new_inst_file_tag)
#         if not osp.isfile(new_inst_lbl_file):
#             assert osp.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)
#             print('Creating {} from {}'.format(new_inst_lbl_file, inst_lbl_file))
#             convert_to_p_mode_file(inst_lbl_file, new_inst_lbl_file, palette=self.instance_palette,
#                                    assert_inside_palette_range=True)
#         return img_file, new_sem_lbl_file, inst_lbl_file
#
#     def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
#         old_sem_lbl_file = sem_lbl_file.replace(self.new_sem_file_tag, self.old_sem_file_tag)
#         old_inst_lbl_file = inst_lbl_file.replace(self.new_inst_file_tag, self.old_inst_file_tag)
#         assert osp.isfile(old_sem_lbl_file)
#         return img_file, old_sem_lbl_file, old_inst_lbl_file
#
#
