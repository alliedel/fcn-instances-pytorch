import numpy as np

from instanceseg.utils import datasets
from instanceseg.datasets.instance_dataset import InstanceDatasetBase, TransformedInstanceDataset
from instanceseg.datasets import coco_format

BACKGROUND_WHITE = (255, 255, 255)
BLUE_RGB = (0, 0, 224)
GREEN_RGB = (0, 225, 0)
PEPTO_BISMOL_PINK_RGB = (246, 143, 224)

ALL_BLOB_CLASS_NAMES = np.array(['background', 'square', 'circle'])


class Defaults(object):
    img_size = (281, 500)
    blob_size = (40, 40)
    clrs = (BACKGROUND_WHITE, BLUE_RGB, GREEN_RGB, PEPTO_BISMOL_PINK_RGB)
    blob_types = ['square', 'circle']
    n_instances_per_img = 2
    return_torch_type = False
    location_generation_type = 'random'  # 'moving'
    velocity_r_c = [[0, 1], [0, -1]]
    n_images = 100
    mean_bgr = np.array([10.0, 10.0, 10.0])
    transform = True
    portrait = False


class BlobExampleGenerator(InstanceDatasetBase):
    def __init__(self, img_size=Defaults.img_size, blob_size=Defaults.blob_size,
                 clrs=Defaults.clrs,
                 n_instances_per_img=Defaults.n_instances_per_img,
                 return_torch_type=Defaults.return_torch_type,
                 n_images=None, mean_bgr=None,
                 transform=Defaults.transform,
                 one_dimension=None,
                 semantic_subset_to_generate=None,
                 ordering=None,
                 _im_a_copy=False,
                 intermediate_write_path='/tmp/',
                 portrait=Defaults.portrait,
                 random_seed=None
                 ):
        """
        one_dimension: {'x', 'y', None}
        portrait: rotates the image size (flips order)
        """
        if mean_bgr is None:
            mean_bgr = Defaults.mean_bgr
        if blob_size is None:
            blob_size = Defaults.blob_size
        n_images = n_images or Defaults.n_images
        if semantic_subset_to_generate is not None:
            assert all([cls_name in ALL_BLOB_CLASS_NAMES for cls_name in semantic_subset_to_generate]), ValueError(
                'semantic_subset={} is incorrect. Must be a list of semantic classes in {}'.format(
                    semantic_subset_to_generate, ALL_BLOB_CLASS_NAMES))
            assert any([cls_name == 'background' for cls_name in semantic_subset_to_generate]), ValueError(
                'Please do include background in the list of classes.')

        assert one_dimension in [None, 'x', 'y'], ValueError
        self.img_size = img_size or Defaults.img_size
        if portrait:
            self.img_size = (self.img_size[1], self.img_size[0])
        assert len(self.img_size) == 2
        self.blob_size = blob_size
        self.clrs = clrs  # nested list: outer list -- one item per semantic class.  inner list
        # -- list of colors for that semantic class.
        self.n_instances_per_img = n_instances_per_img
        self.return_torch_type = return_torch_type
        self.n_images = n_images
        self.mean_bgr = mean_bgr
        self._transform = transform
        self.semantic_subset = semantic_subset_to_generate
        # TODO(allie): change the line below to allow dif. blob types
        self.semantic_classes = semantic_subset_to_generate or ALL_BLOB_CLASS_NAMES
        self.class_names, self.idxs_into_all_blobs = datasets.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset_to_generate, full_set=ALL_BLOB_CLASS_NAMES)
        self.n_instances_per_sem_cls = [0] + [n_instances_per_img for _ in range(len(self.semantic_classes) - 1)]
        self.ordering = ordering.lower() if ordering is not None else None
        self.one_dimension = one_dimension
        self.n_max_per_class = self.n_instances_per_img

        # Blob dynamics
        self.location_generation_type = Defaults.location_generation_type
        if self.location_generation_type == 'random':
            self.random_rows = np.nan * np.ones((self.n_images, len(self.semantic_classes), self.n_max_per_class))
            self.random_cols = np.nan * np.ones((self.n_images, len(self.semantic_classes), self.n_max_per_class))
            self.n_images = n_images
            self.initialize_locations_per_image(random_seed=random_seed)
        else:
            self.n_images = n_images

    def get_semantic_color(self, semantic_idx):
        return self.clrs[semantic_idx]

    @property
    def labels_table(self):
        categories = []
        for sem_idx, cls_name in enumerate(self.semantic_class_names):
            categories.append(coco_format.CategoryCOCOFormat(id=sem_idx,
                                                             name=cls_name,
                                                             color=self.get_semantic_color(sem_idx),
                                                             supercategory=cls_name,
                                                             isthing=cls_name != 'background'))
        return coco_format.create_labels_table_from_list_of_labels(categories)

    def initialize_locations_per_image(self, random_seed=None):
        # initialize to nan to be sure we clear them (for debugging purposes)
        if random_seed:
            np.random.seed(random_seed)
        self.random_rows[:] = np.nan
        self.random_cols[:] = np.nan

        for sem_idx, _ in enumerate(self.semantic_classes):
            n_inst_this_sem_cls = self.n_instances_per_sem_cls[sem_idx]
            if n_inst_this_sem_cls > 0:
                if self.one_dimension == 'x':
                    self.random_rows[:, sem_idx, :n_inst_this_sem_cls] = \
                        np.ones((self.n_images, n_inst_this_sem_cls)) * (self.img_size[0] // 2)
                else:
                    self.random_rows[:, sem_idx, :n_inst_this_sem_cls] = \
                        np.random.randint(0, self.img_size[0] - self.blob_size[0], (self.n_images, n_inst_this_sem_cls))

                if self.one_dimension == 'y':
                    random_cols = np.ones((self.n_images, n_inst_this_sem_cls), dtype=int) * (self.img_size[1] // 2)
                else:
                    random_cols = np.random.randint(0, self.img_size[1] - self.blob_size[1],
                                                    (self.n_images, n_inst_this_sem_cls))

                if self.ordering == 'lr':
                    random_cols.sort(axis=1)
                elif self.ordering is None:
                    pass
                else:
                    raise ValueError('Didn\'t recognize ordering {}'.format(self.ordering))
                self.random_cols[:, sem_idx, :n_inst_this_sem_cls] = random_cols
                # TODO(allie): check for overlap and get rid of it.

    def __getitem__(self, image_index):
        img, (sem_lbl, inst_lbl) = self.generate_img_lbl_pair(image_index)
        return img, (sem_lbl, inst_lbl)

    def __len__(self):
        return self.n_images

    @property
    def semantic_class_names(self):
        return self.class_names

    def copy(self, modified_length=10):
        my_copy = BlobExampleGenerator(_im_a_copy=True)
        for attr, val in self.__dict__.items():
            setattr(my_copy, attr, val)
        assert modified_length <= len(my_copy), "Can\'t create a copy with more examples than " \
                                                "the initial dataset"
        my_copy.n_images = modified_length
        assert len(my_copy) == modified_length
        return my_copy

    def get_blob_coordinates(self, image_index, semantic_idx, instance_idx=None):
        n_instances_in_this_sem_cls = self.n_instances_per_sem_cls[semantic_idx]
        assert instance_idx is None or n_instances_in_this_sem_cls > instance_idx, \
            ValueError('semantic class {} only has {} instances allocated'.format(
                semantic_idx, n_instances_in_this_sem_cls))
        if self.location_generation_type == 'random':
            if instance_idx is None:
                instance_rows = [self.random_rows[image_index, semantic_idx, i]
                                 for i in range(n_instances_in_this_sem_cls)]
                instance_cols = [self.random_cols[image_index, semantic_idx, i]
                                 for i in range(n_instances_in_this_sem_cls)]
                return instance_rows, instance_cols
            else:
                r = self.random_rows[image_index, semantic_idx, instance_idx]
                c = self.random_cols[image_index, semantic_idx, instance_idx]
                return r, c
        else:
            raise ValueError

    def generate_img_lbl_pair(self, image_index):
        img = np.zeros(self.img_size + (3,), dtype=float)
        sem_lbl = np.zeros(self.img_size, dtype=int)
        inst_lbl = np.zeros(self.img_size, dtype=int)

        for semantic_idx, semantic_class in enumerate(self.semantic_classes):
            for instance_idx in range(self.n_instances_per_sem_cls[semantic_idx]):
                r, c = self.get_blob_coordinates(image_index, semantic_idx, instance_idx=instance_idx)
                instance_id = instance_idx + 1

                if semantic_class == 'square':
                    img = self.paint_my_square_in_img(img, r, c, self.clrs[semantic_idx])
                    sem_lbl = self.paint_my_square_in_lbl(sem_lbl, r, c, semantic_idx)
                    inst_lbl = self.paint_my_square_in_lbl(inst_lbl, r, c, instance_id)
                elif semantic_class == 'circle':
                    img = self.paint_my_circle_in_img(img, r, c, self.clrs[semantic_idx])
                    sem_lbl = self.paint_my_circle_in_lbl(sem_lbl, r, c, semantic_idx)
                    inst_lbl = self.paint_my_circle_in_lbl(inst_lbl, r, c, instance_id)
                else:
                    raise ValueError('I don\'t know how to draw {}'.format(semantic_class))
        return img, (sem_lbl, inst_lbl)

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
        r, c = int(r), int(c)
        return paint_square(img, r, c, self.blob_size[0], self.blob_size[1], clr,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=2)

    def paint_my_square_in_lbl(self, lbl_img, r, c, instance_label):
        # noinspection PyTypeChecker
        r, c = int(r), int(c)
        return paint_square(lbl_img, r, c, self.blob_size[0], self.blob_size[0], instance_label,
                            allow_overflow=True, row_col_dims=(0, 1), color_dim=None)


class TransformedBlobExampleGenerator(TransformedInstanceDataset):
    def __init__(self, raw_synthetic_dataset, precomputed_file_transformation=None, runtime_transformation=None):
        super(TransformedBlobExampleGenerator, self).__init__(raw_synthetic_dataset, precomputed_file_transformation,
                                                              runtime_transformation)


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
    in_ellipse = ((c - center_c) / a) ** 2 + ((r - center_r) / b) ** 2 <= 1
    return in_ellipse

# def draw_ellipse_on_pil_img(img, start_r, start_c, end_r, end_c, clr):
#     draw = ImageDraw.Draw(image)
#     draw.ellipse((start_c, start_r, end_c, end_r), fill=clr,
#                  outline=clr)
#     return image
