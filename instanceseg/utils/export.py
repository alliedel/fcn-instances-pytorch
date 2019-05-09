import numpy as np
from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter


def convert_mpl_to_np(figure_handle):
    figure_handle.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(figure_handle.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure_handle.canvas.get_width_height()[::-1] + (3,))
    return data


def log_images(writer: SummaryWriter, tag, images, step, numbers=None, bgr=False):
    assert type(images[0]) is np.ndarray
    if numbers is None:
        numbers = range(len(images))
    for nr, img in enumerate(images):
        if writer is not None:
            writer.add_image('%s/%d' % (tag, numbers[nr]), (img.astype(float) /
                             255.0), global_step=step, dataformats='HWC')


def log_plots(writer: SummaryWriter, tag, plot_handles, step, numbers=None):
    """Logs a list of images."""
    if numbers is None:
        numbers = range(len(plot_handles))
    assert len(numbers) == len(plot_handles), 'len(plot_handles): {}; numbers: {}'.format(len(
        plot_handles), numbers)
    for nr, plot_handle in enumerate(plot_handles):
        # Write the image to a string
        h = plt.figure(plot_handle.number)
        plt_as_np_array = convert_mpl_to_np(h)

        # Create an Image object
        if writer is not None:
            writer.add_image('%s/%d' % (tag, numbers[nr]), plt_as_np_array, global_step=step,
                             dataformats='HWC')  # APD: havent yet confirmed this is correct
