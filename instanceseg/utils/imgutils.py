import PIL.Image
from PIL import Image
import numpy as np


def resize_np_img(img, sz_hw, resample_mode='nearest'):
    resample_opts = {
            'nearest': Image.NEAREST,
            'antialias': Image.ANTIALIAS,
            'bilinear': Image.BILINEAR
        }
    try:
        resample = resample_opts[resample_mode]
    except KeyError:
        raise ValueError('resample_mode ({}) must be one of {}'.format(resample_mode, resample_opts))
    c3 = img.shape[2] if len(img.shape) == 3 else None
    dtype = img.dtype
    if len(sz_hw) == 2:
        sz = (sz_hw[1], sz_hw[0])
    else:
        sz = (sz_hw[1], sz_hw[0], sz_hw[2])
    if img.dtype != 'uint8':
        try:
            assert np.all(img.astype('uint8') == img)
        except:
            import ipdb; ipdb.set_trace()
            raise
        img = img.astype('uint8')
    img = np.array(Image.fromarray(img).resize(sz, resample=resample)).astype(dtype)
    if c3 is not None:
        assert img.shape[2] == c3
    assert img.shape[:2] == sz_hw[:2], 'Debug error: Resize was supposed to create shape {}, but created shape {} ' \
                                       'instead'.format(sz_hw, img.shape)
    return img


def get_new_size(img, multiplier):
    h, w = img.shape[:2]
    h, w = int(multiplier * h), int(multiplier * w)
    return h, w


def resize_img_by_multiplier(img, multiplier):
    dtype = img.dtype
    h, w = get_new_size(img, multiplier)
    img = resize_np_img(img, (h, w)).astype(dtype)
    return img


def load_img_as_dtype(img_file, dtype=None):
    img = PIL.Image.open(img_file)
    if dtype is not None:
        img = np.array(img, dtype=dtype)
    return img


def write_np_array_as_img(arr, filename):
    im = PIL.Image.fromarray(arr.astype(np.uint8))
    im.save(filename)