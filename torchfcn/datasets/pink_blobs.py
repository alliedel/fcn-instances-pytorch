import numpy as np
import torch
from torchfcn.datasets import dataset_utils
from PIL import Image, ImageDraw

PEPTO_BISMOL_PINK_RGB = (246, 143, 224)
BLUE_RGB = (0, 0, 224)
GREEN_RGB = (0, 225, 0)

ALL_BLOB_CLASS_NAMES = np.array(['background', 'pink_square', 'blue_square'])


class Defaults(object):
    img_size = (281, 500)
    blob_size = (40, 40)
    clrs = [PEPTO_BISMOL_PINK_RGB, BLUE_RGB, GREEN_RGB]
    blob_types = ['square', 'circle']
    n_max_per_class = 3
    n_instances_per_img = 2
    return_torch_type = False
    location_generation_type = 'random'  # 'moving'
    velocity_r_c = [[0, 1], [0, -1]]
    max_index = 100
    mean_bgr = np.array([10.0, 10.0, 10.0])
    transform = True


class BlobExampleGenerator(object):
    def __init__(self, img_size=Defaults.img_size, blob_size=Defaults.blob_size, clrs=Defaults.clrs,
                 n_max_per_class=Defaults.n_max_per_class,
                 n_instances_per_img=Defaults.n_instances_per_img,
                 return_torch_type=Defaults.return_torch_type,
                 max_index=Defaults.max_index, mean_bgr=Defaults.mean_bgr,
                 transform=Defaults.transform, velocity_r_c=None,
                 initial_rows=None, initial_cols=None):
        self.img_size = img_size
        assert len(self.img_size) == 2
        self.blob_size = blob_size
        self.clrs = clrs  # nested list: outer list -- one item per semantic class.  inner list
        # -- list of colors for that semantic class.
        self.n_instances_per_img = n_instances_per_img
        self.n_max_per_class = n_max_per_class
        self.return_torch_type = return_torch_type
        self.max_index = max_index
        self.mean_bgr = mean_bgr
        self._transform = transform
        self.semantic_subset = None
        # TODO(allie): change the line below to allow dif. blob types
        self.semantic_classes = Defaults.blob_types
        self.class_names, self.idxs_into_all_blobs = self.get_semantic_names_and_idxs(
            self.semantic_subset)

        # Blob dynamics
        self.location_generation_type = Defaults.location_generation_type
        if self.location_generation_type == 'moving':
            self.velocity_r_c = Defaults.velocity_r_c if velocity_r_c is None else velocity_r_c
            if self.n_instances_per_img != 2:
                assert NotImplementedError('Gotta pick initial conditions (random?)')
            else:
                self.initial_rows = initial_rows if initial_rows is not None else \
                    [0, self.img_size[0] - self.blob_size[0]]
                self.initial_cols = initial_cols if initial_cols is not None else \
                    [0, self.img_size[1] - self.blob_size[1]]
            assert len(self.velocity_r_c) == self.n_instances_per_img, ValueError
            assert len(self.velocity_r_c[0]) == 2, ValueError
            max_index_r_c = [max_index if abs(self.velocity_r_c[0][j]) == 0 else
                             np.floor((self.img_size[j] - self.blob_size[j]) /
                                      abs(self.velocity_r_c[0][j])) for j in [0, 1]]
            self.max_index = np.min(max_index_r_c)
        elif self.location_generation_type == 'random':
            self.max_index = max_index
            n_instance_classes = self.n_max_per_class * len(self.semantic_classes)
            self.random_rows = np.random.randint(0, self.img_size[0] - self.blob_size[
                0], (max_index + 1, n_instance_classes))
            self.random_cols = np.random.randint(0, self.img_size[1] - self.blob_size[
                1], (max_index + 1, n_instance_classes))
            # TODO(allie): check for overlap and get rid of it.

        else:
            self.max_index = max_index

    def __getitem__(self, image_index):
        img, instance_lbl = self.generate_img_lbl_pair(image_index)
        if self._transform:
            img, instance_lbl = self.transform(img, instance_lbl)
        return img, instance_lbl

    def __len__(self):
        return self.max_index + 1

    def get_blob_coordinates(self, image_index, instance_idx=None):
        if self.location_generation_type == 'moving':
            if instance_idx is not None:
                r = self.initial_rows[instance_idx] + self.velocity_r_c[instance_idx][0] * image_index
                c = self.initial_cols[instance_idx] + self.velocity_r_c[instance_idx][1] * image_index
                return r, c
            instance_rows = [self.initial_rows[i] + self.velocity_r_c[i][0] * image_index
                             for i, start_row in enumerate(self.initial_rows)]
            instance_cols = [start_col + self.velocity_r_c[i][1] * image_index
                             for i, start_col in enumerate(self.initial_cols)]
            return instance_rows, instance_cols
        elif self.location_generation_type == 'random':
            if instance_idx is not None:
                r = self.random_rows[image_index, instance_idx]
                c = self.random_cols[image_index, instance_idx]
                return r,c
            instance_rows = [self.random_rows[image_index, i] for i in range(
                self.n_instances_per_img)]
            instance_cols = [self.random_cols[image_index, i] for i in range(
                self.n_instances_per_img)]
            return instance_rows, instance_cols
        else:
            raise ValueError

    def generate_img_lbl_pair(self, image_index):
        img = np.zeros(self.img_size + (3,), dtype=float)
        lbl = np.zeros(self.img_size, dtype=int)
        for semantic_idx, semantic_class in enumerate(self.semantic_classes):
            for instance_number in range(self.n_instances_per_img):
                instance_label = semantic_idx * self.n_max_per_class + instance_number + 1
                r, c = self.get_blob_coordinates(image_index, instance_label)
                if semantic_class == 'square':
                    img = self.paint_my_square_in_img(img, r, c, self.clrs[semantic_idx])
                    lbl = self.paint_my_square_in_lbl(lbl, r, c, instance_label)
                elif semantic_class == 'circle':
                    img = self.paint_my_circle_in_img(img, r, c, self.clrs[semantic_idx])
                    lbl = self.paint_my_circle_in_lbl(lbl, r, c, instance_label)
                else:
                    ValueError('I don\'t know how to draw {}'.format(semantic_class))
        if not lbl.max() < self.n_max_per_class * len(self.semantic_classes) + 1:
            import ipdb; ipdb.set_trace()
        return img, lbl

    def paint_my_circle_in_img(self, img, r, c, clr):
        # noinspection PyTypeChecker
        return paint_circle(img, r, c, self.blob_size[0], self.blob_size[1], clr,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=2)

    def paint_my_circle_in_lbl(self, lbl_img, r, c, instance_label):
        # noinspection PyTypeChecker
        return paint_circle(lbl_img, r, c, diameter_a=self.blob_size[0],
                            diameter_b=self.blob_size[0],
                            clr=instance_label,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=None)

    def paint_my_square_in_img(self, img, r, c, clr):
        return paint_square(img, r, c, self.blob_size[0], self.blob_size[1], clr,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=2)

    def paint_my_square_in_lbl(self, lbl_img, r, c, instance_label):
        # noinspection PyTypeChecker
        return paint_square(lbl_img, r, c, self.blob_size[0], self.blob_size[0], instance_label,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=None)

    def transform(self, img, lbl):
        img = dataset_utils.transform_img(img, self.mean_bgr)
        lbl = dataset_utils.transform_lbl(lbl)
        return img, lbl

    def untransform(self, img, lbl):
        img = dataset_utils.untransform_img(img, self.mean_bgr)
        lbl = dataset_utils.untransform_lbl(lbl)
        return img, lbl

    @staticmethod
    def get_semantic_names_and_idxs(semantic_subset):
        return dataset_utils.get_semantic_names_and_idxs(semantic_subset=semantic_subset,
                                                         full_set=ALL_BLOB_CLASS_NAMES)


def paint_square(img, start_r, start_c, w, h, clr, allow_overflow=True, row_col_dims=(0, 1),
                 color_dim=2):
    """
    allow_overflow = False: error if square falls off edge of screen.
    row_col_dims, color_dim: set to (2, 3), 1 if you're using pytorch defaults, for instance
    (num_images, channels, rows, cols)
    Other options: (0,1), 2 -- typical image RGB format
                    (0,1), None -- grayscale format (for labels, for instance)
    """

    img_h, img_w = img.shape[0], img.shape[1]

    end_r = start_r + h
    end_c = start_c + w
    if not allow_overflow:
        if end_r > img_h:
            raise Exception('square overflows image.')
        if end_c > img_w:
            raise Exception('square overflows image.')
    else:
        end_r = min(end_r, img_h)
        end_c = min(end_c, img_w)

    # Handle 'flat' images
    if color_dim is None:
        assert np.isscalar(clr), ValueError
        assert row_col_dims == (0, 1), NotImplementedError
        img[start_r:end_r, start_c:end_c] = clr
        return img
    # Handle RGB images
    if row_col_dims == (0, 1):
        for ci, c in enumerate(clr):
            img[start_r:end_r, start_c:end_c, ci] = c
    elif row_col_dims == (2, 3):
        for ci, c in enumerate(clr):
            img[:, ci, start_r:end_r, start_c:end_c] = c
    else:
        raise NotImplementedError
    return img


def paint_circle(img, start_r, start_c, diameter_a, diameter_b, clr, allow_overflow=True,
                 row_col_dims=(0, 1),
                 color_dim=2):
    """
    allow_overflow = False: error if square falls off edge of screen.
    row_col_dims, color_dim: set to (2, 3), 1 if you're using pytorch defaults, for instance
    (num_images, channels, rows, cols)
    Other options: (0,1), 2 -- typical image RGB format
                    (0,1), None -- grayscale format (for labels, for instance)
    """
    img_h, img_w = img.shape[0], img.shape[1]
    end_r = start_r + diameter_b
    end_c = start_c + diameter_a
    if not allow_overflow:
        if end_r > img_h:
            raise Exception('square overflows image.')
        if end_c > img_w:
            raise Exception('square overflows image.')
    else:
        end_r = min(end_r, img_h)
        end_c = min(end_c, img_w)

    a, b = int(diameter_a / 2), int(diameter_b / 2)
    # Handle 'flat' images
    if color_dim is None:
        assert np.isscalar(clr), ValueError
        assert row_col_dims == (0, 1), NotImplementedError
        img_shape = img.shape[:2]
        bool_ellipse_img = valid_ellipse_locations(
            img_shape, center_r=start_r + b, center_c=start_c + a, b=b, a=a)
        img = (1 - bool_ellipse_img) * img + (bool_ellipse_img) * clr
        return img
    # Handle RGB images
    if row_col_dims == (0, 1):
        img_shape = img.shape[:2]
        bool_ellipse_img = valid_ellipse_locations(
            img_shape, center_r=start_r + b, center_c=start_c + a, b=b, a=a)
        bool_indexing = np.tile(bool_ellipse_img[:, :, np.newaxis], (1, 1, 3))
        img = (1 - bool_indexing) * img + (bool_indexing) * clr

    elif row_col_dims == (2, 3):
        raise NotImplementedError
        img_shape = img.shape[2:4]
        bool_ellipse_img = valid_ellipse_locations(
            img_shape, center_r=start_r + b, center_c=start_c + a, b=b, a=a)
        img[np.tile(bool_ellipse_img[np.newaxis, np.newaxis, :, :], (1, 3, 1, 1))] = \
            np.array(clr)[np.newaxis, :, np.newaxis, np.newaxis]
    else:
        raise NotImplementedError
    return img


def valid_ellipse_locations(img_shape, center_r, center_c, b, a):
    r = np.arange(img_shape[0])[:, None]
    c = np.arange(img_shape[1])
    in_ellipse = ((c - center_c)/a)**2 + ((r - center_r)/b)**2 <= 1
    return in_ellipse


def draw_ellipse_on_pil_img(img, start_r, start_c, end_r, end_c, clr):
    draw = ImageDraw.Draw(image)
    draw.ellipse((start_c, start_r, end_c, end_r), fill=clr,
                 outline=clr)
    return image


