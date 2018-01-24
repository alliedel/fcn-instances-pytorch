PEPTO_BISMOL_PINK_RGB = (246, 143, 224)





def paint_square(img, start_r, start_c, w, h, clr, allow_overflow=True, row_col_clr_dims=(0,1,2)):
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
