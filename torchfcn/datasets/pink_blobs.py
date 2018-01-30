import numpy as np

PEPTO_BISMOL_PINK_RGB = (246, 143, 224)


class PinkBlobExampleGenerator():
    def __init__(self, img_size=(60,60), blob_size=(4,4), clr=PEPTO_BISMOL_PINK_RGB,
                 n_max_per_class=3, n_instances_per_img=2, return_torch_type=False):
        self.img_size = img_size
        self.blob_size = blob_size
        self.clr = clr
        self.n_instances_per_img = n_instances_per_img
        self.n_max_per_class = n_max_per_class
        self.return_torch_type = return_torch_type

        assert len(self.img_size) == 2

    def generate_img_lbl_pair(self):
        img = np.zeros(self.img_size + (3,))
        lbl = np.zeros(self.img_size + (self.n_max_per_class,))

        paint_my_square_in_lbl = lambda lbl, r, c: paint_square(lbl, r, c, square_size[0],
                                                            square_size[1], clr,
                                                            allow_overflow=True,
                                                            row_col_clr_dims=(0, 1, 2))

    def paint_my_square_in_img(img, r, c):
            return paint_square(img, r, c, square_size[0], square_size[1],
                                clr, allow_overflow=True, row_col_clr_dims=(0, 1, 2))
    start_rows = [0, img_size[0] - square_size[0]]
    start_cols = [0, img_size[1] - square_size[1]]
    for square_idx, (r, c) in enumerate(zip(start_rows, start_cols)):
        img = paint_my_square_in_img(img, r, c)
        lbl = paint_my_square_in_img(img, r, c)


def paint_square(img, start_r, start_c, w, h, clr, allow_overflow=True, row_col_clr_dims=(0, 1, 2)):
    """
    allow_overflow = False: error if square falls off edge of screen.
    row_col_clr_dims: set to (2, 3, 1) if you're using pytorch defaults, for instance (
    num_images, channels, rows, cols)
    """

    end_r = start_r + h
    end_c = start_c + w
    if not allow_overflow:
        if end_r > h:
            raise Exception('square overflows image.')
        if end_c > w:
            raise Exception('square overflows image.')
    else:
        end_r = min(end_r, h)
        end_c = min(end_c, w)
    if row_col_clr_dims == (0,1,2):
        for ci, c in enumerate(clr):
            img[start_r:end_r, start_c:end_c, ci] = c
    elif row_col_clr_dims == (0,1,2):
        for ci, c in enumerate(clr):
            img[:, ci, start_r:end_r, start_c:end_c] = c
    else:
        raise NotImplementedError('Haven\'t implemented that row_col_clr_dims option: {}'.format(
            row_col_clr_dims))
    return img
