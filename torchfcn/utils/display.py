import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from textwrap import wrap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torchfcn.utils.misc import TermColors
import cycler

WORKSPACE_DIR = os.path.expanduser('~/workspace/images')

GOOD_COLOR_CYCLE = ['#3B444B', '#E52B50', '#4B5320', '#FFE135', '#BF94E4',
                    '#FF2052']

GOOD_COLORS = ['#332288', '#117733', '#DDCC77', '#88CCEE', '#CC6677',
               '#882255', '#AA4499', '#44AA99', '#999933']

GOOD_COLORS_BY_NAME = {
    'aqua': '#44AA99',
    'b': '#332288',
    'blue': '#332288',
    'default': '#3B444B',
    'darkgrey': '#3B444B',
    'k': '#3B444B',
    'g': '#117733',
    'green': '#117733',
    'lightblue': '#88CCEE',
    'r': '#E52B50',
    'red': '#E52B50',
    'v': '#882255',
    'violet': '#882255',
    'y': '#DDCC77',
    'yellow': '#DDCC77'
}


MY_RC_DEFAULTS = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 10,
    'font.size': 8, # was 10
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    'figure.figsize': [3.39, 2.10],
    'font.family': 'serif',
    'lines.linewidth': 3,
    'axes.prop_cycle': cycler.cycler('color', GOOD_COLOR_CYCLE),
}

# MARKERS = mpl.markers.MarkerStyle.filled_markers[::-1]
MARKERS = ('o', 'X', '*', 'v', 's', 'p', 'h', 'H', 'D', 'd', 'P', '8')

DPI = 300


def check_for_emptied_workspace(workspace_dir=WORKSPACE_DIR, interactive=True):
    if len(os.listdir(workspace_dir)) == 0:
        return True

    override = False
    if interactive:
        override = 'y' == input(
            TermColors.WARNING + 'Your workspace isn\'t clean.  Would you like to continue anyway? [y/N]\n ' +
            TermColors.ENDC)
        if override:
            return False
    raise Exception(TermColors.FAIL + 'Exiting.  Please run the following command: clear_workspace.'
                    + TermColors.ENDC)


def my_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def get_color(i=0):
    return GOOD_COLORS[i % len(GOOD_COLORS)]


def get_marker(i):
    return MARKERS[i % len(MARKERS)]


def get_subplot_idx(r, c, shape):
    C = shape[1]
    return C * r + c + 1


def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind.astype('int')


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def subplot_grid(num_rows, num_cols):
    subplot_grid = gridspec.GridSpec(num_rows, num_cols)
    return subplot_grid


def nested_subplot_grid(num_rows, num_cols, parent_grid):
    nested_grid = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols, subplot_spec=parent_grid)
    return nested_grid


def scatter3(x, y, z, ax=None, zdir='z', **plot_args):
    # plot_args examples: c='r', marker='x', label='label'
    if ax is None:
        ax = Axes3D(plt.gcf())
    ax.scatter3D(x, y, z, zdir=zdir, **plot_args)
    return ax


def stem(x, yval, linefmt='b-', label=None, width=None, color=GOOD_COLORS[0], **kwargs):
    if x.dtype == 'bool':
        y = np.ones(np.sum(x)) * yval
        x = np.where(x)[0]
    else:
        y = np.ones(len(x)) * yval
    (markerline, stemlines, baseline) = plt.stem(x[:], y[:], linefmt=linefmt, markerfmt='b',
                                                 basefmt='k', label=label, **kwargs)
    if width:
        plt.setp(stemlines, 'linewidth', width)
    if color:
        plt.setp(stemlines, 'color', color)
        plt.setp(markerline, 'color', color)
    plt.setp(baseline, visible=False)
    plt.setp(markerline, visible=False)


def display_training_sets(training_sets, figure_numbers=None):
    if figure_numbers is None:
        figure_numbers = range(len(training_sets))
    for set_index, training_set in enumerate(training_sets):
        plt.figure(figure_numbers[set_index])
        display_list_of_images(training_set)


def get_good_clims_for_arr(arr, std_multiplier=4.0):
    """
    perc_in_range: Minumum percentage of points that must fall in the clim range
    """
    my_min = np.min(arr)
    my_max = np.max(arr)
    my_mean = np.mean(arr)
    my_std = np.std(arr)

    clim_max = min(my_max, my_mean + (std_multiplier * my_std))
    clim_min = max(my_min, my_mean - (std_multiplier * my_std))

    perc_in_range = np.mean(np.logical_and(arr >= clim_min, arr <= clim_max))
    if perc_in_range < 0.95:
        print('clim range ({:02f}, {:02f}) only contains {:02f}% of the data.'.format(clim_min, clim_max,
                                                                                   100 * perc_in_range))
    return clim_min, clim_max


def display_list_of_images(list_of_images, list_of_titles=None, arrange='square', arrangement_rc_list=None,
                           sync_clims=True, show_clim='alone', cmap='gray', smart_clims=False):
    """
    arrangement_rc_list : [(r1, c1), (r2, c2), (r_n, c_n)] for each image in list_of_images (n of them)
    smart_clims: by stddev
    """
    arrange_options = ['rows', 'cols', 'square', 'custom']
    assert any([arrange == option for option in arrange_options])
    assert not (show_clim == 'alone' and not sync_clims), ValueError(
        'show_clim=\'alone\' implies that sync_clims should be set to True.')

    fig = plt.gcf()

    if smart_clims:
        smart_clims_per_img = [get_good_clims_for_arr(img) for img in list_of_images]
        clims = (min([cl[0] for cl in smart_clims_per_img]), max([cl[1] for cl in smart_clims_per_img]))
    else:
        clims = None
    if arrange == 'rows':
        R = len(list_of_images)
        C = 1
        arrangement_rc_list = [(r, 0) for r in range(R)]
    elif arrange == 'cols':
        C = len(list_of_images)
        R = 1
        arrangement_rc_list = [(0, c) for c in range(C - 1)]
    elif arrange == 'square':  # TODO(allie): Debug this arrange mode (something was going wrong w/ colorbar earlier)
        R = np.floor(np.sqrt(len(list_of_images)))
        C = np.ceil(len(list_of_images) / float(R))
        arrangement_rc_list = [divmod(ind, C) for ind in range(len(list_of_images))]
    elif arrange == 'custom':
        assert arrangement_rc_list is not None and len(arrangement_rc_list) == len(list_of_images) and \
               all([len(rc) == 2 for rc in arrangement_rc_list]), \
            ValueError('Please format arrangement_rc_list correctly')
        R = max([rc[0] for rc in arrangement_rc_list]) + 1
        C = max([rc[1] for rc in arrangement_rc_list]) + 1
    for image_index, image in enumerate(list_of_images):
        rc = arrangement_rc_list[image_index]
        subplot_index = get_subplot_idx(r=rc[0], c=rc[1], shape=(R, C))
        plt.subplot(R, C, subplot_index)
        matshow(image, show_colorbar=False, cmap=cmap)
        plt.set_cmap(cmap)
        if clims is not None:
            plt.clim(*clims)
        if not sync_clims:
            plt.colorbar()
        if list_of_titles:
            plt.title(list_of_titles[image_index])
        else:
            plt.title(str(image_index))
    plt.tight_layout()
    if show_clim == 'alone':
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.05, 0.15, 0.85])
        plt.sca(cbar_ax)
        plt.set_cmap(cmap)
        clims = plt.gci().get_clim()
        mappable = cbar_ax.imshow(np.array([clims]))
        plt.gca().set_visible(False)
        cbar = plt.colorbar(mappable, ax=cbar_ax, fraction=0.95)
    if sync_clims:
        sync_clim_axes(plt.gcf().axes)


def mnist_vector_as_image(vector, shape=(28, 28), transpose=True):
    """
    If no shape is specified, assumes a square image.
    """
    if transpose:
        return np.reshape(vector, shape).T
    else:
        return np.reshape(vector, shape)


def display_vector_as_image(vector, shape, title='', **kwargs):
    h = matshow(np.reshape(vector, shape), **kwargs)
    h.axes.get_xaxis().set_ticks([])
    h.axes.get_yaxis().set_ticks([])
    plt.title(title)


def sync_clim_fignums(fignums, clim=()):
    """
    Finds the min, max of each of the axes, and sets the clim for all of them to be the same.
    """
    ax_list = []
    for fignum in fignums:
        h = plt.figure(fignum)
        ax_list += h.axes
    sync_clim_axes(ax_list, clim=clim)


def sync_clim_axes(axes_list, clim=()):
    if len(clim) == 0:
        cmin = np.inf
        cmax = -np.inf
        for ax in axes_list:
            plt.sca(ax)
            clims = plt.gci().get_clim()
            cmin = min(clims[0], cmin)
            cmax = max(clims[1], cmin)
        if cmin == cmax:
            cmax = cmin + 1e-3
    else:
        cmin = clim[0]
        cmax = clim[1]
    for ax in axes_list:
        plt.sca(ax)
        plt.clim(cmin, cmax)


def matshow(mat, aspect=None, show_colorbar=True, cmap='gray'):
    """ Uses imshow with gray colormap; adds a colorbar; and uses nearest interpolation"""
    if aspect is None:
        h = plt.imshow(mat, interpolation='nearest')
    else:
        h = plt.imshow(mat, aspect=aspect)
    plt.set_cmap(cmap)
    if show_colorbar:
        plt.colorbar()

    h.axes.get_xaxis().set_ticks([])
    h.axes.get_yaxis().set_ticks([])
    return h


def imwrite_to_workspace(img, filename, workspace_dir=WORKSPACE_DIR, cmap=None, **imsave_args):
    cmap = cmap if cmap is not None else matplotlib.rc_params()['image.cmap']
    plt.imsave(fname=os.path.join(workspace_dir, filename), arr=img, cmap=cmap, **imsave_args)


def imwrite_list_as_2d_array_to_workspace(img_list_2d, filename_base_ext='.png',
                                          imshape=None, workspace_dir=WORKSPACE_DIR, **imsave_args):
    # Each column is an image; we'll reshape each image to imshape and then write.
    if imshape is None:
        imshape = (img_list_2d.shape[0], 1)
    img_list = []
    for im in img_list_2d:
        img_list.append(np.reshape(im, imshape))
    imwrite_list_to_workspace(img_list, filename_base_ext=filename_base_ext,
                              workspace_dir=workspace_dir, **imsave_args)


def imwrite_list_as_3d_array_to_workspace(img_list_3d, filename_base_ext='.png',
                                          workspace_dir=WORKSPACE_DIR, **imsave_args):
    # Each img_list_3d[:,:,i] is an image; we'll write each individually.
    img_list = []
    for im_idx in range(img_list_3d.shape[2]):
        img_list.append(img_list_3d[:, :, im_idx])
    imwrite_list_to_workspace(img_list, filename_base_ext=filename_base_ext,
                              workspace_dir=workspace_dir, **imsave_args)


def imwrite_list_to_workspace(img_list, filename_base_ext='.png', workspace_dir=WORKSPACE_DIR,
                              **imsave_args):
    # img_list can either be a list of arrays or a 3D array (in which case, the 3rd dimension
    # will be assumed to iterate over the list of images; e.g. img_list[:,:,0] is the first image).
    # Ex: filename_base_ext = 'eigenvalues.png' --> Writes each image to eigenvalues_000001.png,
    # eigenvalues_000002.png, etc.
    filename_base, ext = os.path.splitext(filename_base_ext)
    if not ext and filename_base and filename_base[0] == '.':
        ext = filename_base
        filename_base = ''

    filenames = []
    for img_idx in range(len(img_list)):
        filename = os.path.join(workspace_dir, filename_base + '{0:06}'.format(img_idx) + ext)
        imwrite_to_workspace(img_list[img_idx], filename, workspace_dir=workspace_dir,
                             **imsave_args)
        filenames.append(filename)
    return filenames


def matshow_and_save_3d_array_to_workspace(mats_as_3d_array, filename_base_ext='.png', aspect=None,
                                           show_colorbar=True, sync_clims=True, set_clims=(),
                                           workspace_dir=WORKSPACE_DIR, list_of_titles=None,
                                           show_filenames_as_titles=False, cmap='gray'):
    """
    mats_as_3d_array: set of 2d arrays stacked in the 3rd dimension
    """
    try:
        assert not (show_filenames_as_titles and (list_of_titles is not None))
    except:
        import ipdb;
        ipdb.set_trace()
        raise
    matshow_and_save_list_to_workspace([mats_as_3d_array[:, :, i] for i in range(mats_as_3d_array.shape[2])],
                                       filename_base_ext=filename_base_ext, aspect=aspect,
                                       show_colorbar=show_colorbar, sync_clims=sync_clims, set_clims=set_clims,
                                       workspace_dir=workspace_dir, show_filenames_as_titles=show_filenames_as_titles,
                                       list_of_titles=list_of_titles, cmap=cmap)


def matshow_and_save_list_to_workspace(list_of_mats, filename_base_ext='.png', aspect=None,
                                       show_colorbar=True, sync_clims=True, set_clims=(),
                                       workspace_dir=WORKSPACE_DIR, list_of_titles=None,
                                       show_filenames_as_titles=False, cmap='gray', **savefig_kwargs):
    """ Uses imshow with gray colormap; adds a colorbar; and uses nearest interpolation"""
    single_image = len(list_of_mats) == 1
    try:
        assert not (show_filenames_as_titles and (list_of_titles is not None))
    except:
        import ipdb;
        ipdb.set_trace()
        raise
    filename_base, ext = os.path.splitext(filename_base_ext)
    if not ext and filename_base and filename_base[0] == '.':
        ext = filename_base
        filename_base = ''

    filenames = []
    fignums = []
    for img_idx in range(len(list_of_mats)):
        filename = os.path.join(workspace_dir, filename_base + ext) if \
            single_image else os.path.join(workspace_dir, filename_base + '{0:06}'.format(img_idx) + ext)
        h = plt.figure()
        matshow(list_of_mats[img_idx], show_colorbar=show_colorbar, cmap=cmap)
        if show_filenames_as_titles:
            ttl = os.path.basename(filename)
            ttl = "\n".join(wrap(ttl, 60))
            plt.title(ttl)
        elif list_of_titles is not None:
            plt.title(list_of_titles[img_idx])
        filenames.append(filename)
        fignums.append(h.number)

    if len(set_clims) > 0:
        sync_clim_fignums(fignums, set_clims)
    elif sync_clims:
        sync_clim_fignums(fignums)

    for fignum, filename in zip(fignums, filenames):
        plt.figure(fignum)
        save_fig_to_workspace(filename, workspace_dir=workspace_dir, **savefig_kwargs)

    return filenames


def save_fig_to_workspace(filename=None, workspace_dir=WORKSPACE_DIR, **savefig_kwargs):
    if not filename:
        filename = '%02d.png' % plt.gcf().number
    plt.savefig(os.path.join(workspace_dir, filename), **savefig_kwargs)


def save_all_figs_to_workspace():
    for i in plt.get_fignums():
        plt.figure(i)
        save_fig_to_workspace()


def set_my_rc_defaults(params=MY_RC_DEFAULTS):
    matplotlib.rcParams.update(params)


def set_latex_default():
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    matplotlib.rc('text', usetex=True)


def get_rc_color_cycle():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']
    # return matplotlib.rcParams['axes.color_cycle']  # deprecated


def imscale(image):
    image -= image.min()
    image /= image.max()
    return image
