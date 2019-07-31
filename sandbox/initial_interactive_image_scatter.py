import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np;

np.random.seed(42)


def my_hover(event, img_arr, line, fig, xybox, ab, im, x, y):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches() * fig.dpi
        ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
        hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0] * ws, xybox[1] * hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(img_arr[ind])
    else:
        # if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()


def generate_data():
    # Generate data x, y for scatter and an array of images.
    x = np.arange(20)
    y = np.random.rand(len(x))
    arr = np.empty((len(x), 10, 10))
    for i in range(len(x)):
        f = np.random.rand(5, 5)
        arr[i, 0:5, 0:5] = f
        arr[i, 5:, 0:5] = np.flipud(f)
        arr[i, 5:, 5:] = np.fliplr(np.flipud(f))
        arr[i, 0:5:, 5:] = np.fliplr(f)
    return x, y, arr


def make_interactive_plot(x, y, im_arr, xlabel='', ylabel=''):
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, ls="", marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # create the annotations box
    im = OffsetImage(im_arr[0], zoom=5)
    xybox = (50., 50.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event',
                           lambda event: my_hover(event, im_arr, line, fig, xybox, ab, im, x, y))
    plt.show()


if __name__ == '__main__':
    x_, y_, im_arr_ = generate_data()
    # create figure and plot scatter
    make_interactive_plot(x_, y_, im_arr_)
