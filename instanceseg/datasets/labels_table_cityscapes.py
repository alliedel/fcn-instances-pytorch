import PIL.ImagePalette
import PIL.Image
import numpy as np
import six.moves

labels_keys = ['name', 'id', 'train_ids', 'category', 'category_id', 'has_instances',
               'ignore_in_eval', 'color']


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
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
    return cmap


def Label(*args):
    assert len(args) == len(labels_keys)
    return dict(zip(labels_keys, args))


# NOTE(allie): We have to add 1 to the train id's because 0 is reserved for background.

CITYSCAPES_LABELS_TABLE = [
    #       name   id    trainId-1   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

class_names = [class_label['name'] for class_label in CITYSCAPES_LABELS_TABLE]
ids = [class_label['id'] for class_label in CITYSCAPES_LABELS_TABLE]
train_ids = [class_label['train_ids'] for class_label in CITYSCAPES_LABELS_TABLE]
has_instances = [class_label['has_instances'] for class_label in CITYSCAPES_LABELS_TABLE]
colors = [class_label['color'] for class_label in CITYSCAPES_LABELS_TABLE]
ignore_in_eval = [class_label['ignore_in_eval'] for class_label in CITYSCAPES_LABELS_TABLE]
is_void = [class_label['ignore_in_eval'] for class_label in CITYSCAPES_LABELS_TABLE]


def get_rgb_semantic_palette_array(vals, corresponding_rgb_colors, unassigned_rgb_color=(0, 0, 0)):
    # Some quick input checks: make sure vals are legal P mode values; make sure assignments don't overlap in the
    # one-to-many sense.
    # assert all([v >= 0 for v in vals]), 'vals cannot contain negatives: {}'.format([v for v in vals if v < 0])
    assert len(vals) == len(corresponding_rgb_colors)
    sorted_idxs = np.argsort(vals)
    old_val, old_clr = None, None
    for val, clr in zip([vals[i] for i in sorted_idxs], [corresponding_rgb_colors[i] for i in sorted_idxs]):
        if old_val is not None and val == old_val:  # make sure color matches
            assert clr == old_clr, 'value {} is assigned to at least two colors ({}, {})'.format(val, clr, old_clr)
        old_val, old_clr = val, clr
    rgb_palette_array = np.ones((256, 3)) * unassigned_rgb_color
    for val, clr in zip(vals, corresponding_rgb_colors):
        rgb_palette_array[val, :] = clr
    return rgb_palette_array


def convert_palette_array_to_list_for_imagepalette_class(rgb_palette_array):
    palette_list = []
    for i in range(rgb_palette_array.shape[0]):
        palette_list += list(rgb_palette_array[i, :])
    return palette_list


def get_semantic_palette(rgb255=(0, 0, 0)):
    bool_not_255 = [train_id != 255 for train_id in train_ids]
    train_ids_to_map = [train_id for idx, train_id in enumerate(train_ids) if bool_not_255[idx]]
    colors_to_map = [color for idx, color in enumerate(colors) if bool_not_255[idx]]
    palette_array = get_rgb_semantic_palette_array(train_ids_to_map + [255], colors_to_map + [rgb255])
    color_palette = PIL.ImagePalette.ImagePalette(mode='RGB',
                                                  palette=convert_palette_array_to_list_for_imagepalette_class(
                                                      palette_array), size=0)  # NOTE(allie): mode='RGB' or 'P'?
    return color_palette


def get_instance_palette_image(n_colors=256, rgb255=(50, 50, 50), rgb0=(0, 0, 0)):
    palette_array = label_colormap(n_colors).astype(int)
    palette_array[-1, :] = rgb255
    palette_array[0, :] = rgb0
    palette_list = convert_palette_array_to_list_for_imagepalette_class(palette_array)
    im = PIL.Image.new('P', (palette_array.shape[0], 1))
    im.putpalette(palette_list)
    return im


def get_semantic_palette_image(rgb255=(0, 0, 0)):
    bool_not_255 = [train_id != 255 for train_id in train_ids]
    train_ids_to_map = [train_id for idx, train_id in enumerate(train_ids) if bool_not_255[idx]]
    colors_to_map = [color for idx, color in enumerate(colors) if bool_not_255[idx]]
    palette_array = get_rgb_semantic_palette_array(train_ids_to_map + [255], colors_to_map + [rgb255]).astype(int)
    palette_list = convert_palette_array_to_list_for_imagepalette_class(palette_array)
    im = PIL.Image.new('P', (palette_array.shape[0], 1))
    im.putpalette(palette_list)
    return im
