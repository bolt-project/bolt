from numpy import asarray, hstack, arange
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import xlim, ylim, gca, figure, text, subplot


class DisplayArray(object):
    """
    Depicts the shape of an ndarray as a sequence of rectangles
    """
    def __init__(self, width=600, height=200, ax=None):
        self.width = width
        self.height = height
        self.xmax = 0
        self.ymax = 0

        if ax is None:
            self.fig = figure(figsize=(12, 6))
            self.ax = gca()

        else:
            self.ax = ax

    def box(self, x, y, width, height, color):
        """
        Draw a box with the given dimensions.

        Parameters
        ----------
        x : float
            Lower left corner, x-coordinate

        y : float
            Lower left corner, y-coordinate

        width : float
            Width of box

        height : float
            Height of box

        color : str or tuple
            Fill color
        """
        rect = Rectangle((x, y), width, height, fill=True, color=color, ec='white', lw=5)
        self.ax.add_patch(rect)

    def draw(self, shape, cmap=None, scaleby=None, normby=None, label=True):
        """
        Draw a sequence of rectangles representing an array shape.

        Parameters
        ----------
        shape : tuple
            The shape of the array to depict

        cmap : colormap, optional, default=None
            Colormap to use to color rectangles, will use a provided cmap if None

        scaleby : float, optional, default=None
            A factor by which to rescale sizes, useful if comparing two arrays

        normby : float, optional, default=None
            Dimension to normalize by, if None will use minimum

        label : boolean, optional, default=True
            Whether to add a text label with shape
        """
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list('blend', ["#E6550D", "#FDD0A2"])

        sizes, orient, scale = self.init(shape, normby)
        scale = scale * scaleby if scaleby is not None else scale
        left, above, xmax, ymax = (0, 0, 0, 0)

        for i, s in enumerate(sizes):
            shift = s * scale
            clr = cmap(i/float(len(sizes)))
            if orient[i] == 0:
                width, height = (shift, scale)
            else:
                width, height = (scale, shift)
            self.box(left, above, width, height, clr)
            ymax = max(ymax, above + height)
            xmax = max(xmax, left + width)
            if orient[i] == 0:
                left += width
            else:
                above += height

        if label is True:
            self.ax.text(0, -0.1*ymax, self.stringify(shape), fontsize=20)
            self.ax.set_ylim([ymax, -0.2*ymax])
        else:
            self.ax.set_ylim([ymax, 0])

        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_xlim([0, xmax])
        self.xmax = xmax
        self.ymax = ymax

        return self

    def init(self, shape, normby=None):
        """
        Initialize display parameters

        Parameters
        ----------
        shape : tuple
            Shape of the array

        normby : float, optional, default=None
            What to normalize by, if None will use minimum dimensions
        """
        shape = asarray(shape, 'float')
        n = len(shape)

        # normalize dims to 1
        normby = normby if normby is not None else shape.min()
        shape /= normby

        # determine which boxes will be horzizontal vs vertical
        orient = asarray([0 if i % 2 == 0 else 1 for i in range(n)])

        # get dimensions of resulting grid
        ncol = sum(shape[orient == 0])
        nrow = sum(shape[orient == 1])
        if orient[-1] == 0:
            nrow += 1
        if orient[-1] == 1:
            ncol += 2

        # automatically determine scale to fill space
        if ncol > nrow:
            scale = min([self.height / nrow, self.width / ncol])
        else:
            scale = self.height / nrow

        return shape, orient, scale

    @staticmethod
    def stringify(shape):
        return (("%s x " * len(shape)) % shape)[:-3]

class DisplayArrayJoint(object):

    def __init__(self):
        from matplotlib import gridspec
        self.fig = figure(figsize=(10, 5))
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        self.ax1 = subplot(self.gs[0])
        self.fig1 = DisplayArray(ax=self.ax1)
        self.ax2 = subplot(self.gs[1])
        self.fig2 = DisplayArray(ax=self.ax2)

    def draw(self, shape1, shape2):
        """
        Draw rectangles representing two arrays shapes

        Parameters
        ----------
        shape1 : tuple
            The shape of the first array

        shape2 : tuple
            The shape of the second array
        """
        # get minimum across both sets of dimensions
        allmin = min(min(list(shape1)), min(list(shape2)))

        # estimate relative scaling
        _, _, scale1 = self.fig1.init(shape1, normby=allmin)
        _, _, scale2 = self.fig2.init(shape2, normby=allmin)
        if scale1 > scale2:
            factor1, factor2 = (1, scale1/scale2)
        else:
            factor1, factor2 = (scale2/scale1, 1)

        # create the two displays
        cmap = LinearSegmentedColormap.from_list('blend', ["#393B79", "#9C9EDE"])
        self.fig1.draw(shape1, scaleby=factor1, normby=allmin, label=False)
        self.fig2.draw(shape2, cmap=cmap, scaleby=factor2, normby=allmin, label=False)

        # get common axis bounds
        ybound = max(self.fig1.ymax, self.fig2.ymax)
        xbound = max(self.fig1.xmax, self.fig2.xmax)

        # add text labels
        self.fig1.ax.text(0, -0.1*ybound, DisplayArray.stringify(shape1), fontsize=20)
        self.fig2.ax.text(0, -0.1*ybound, DisplayArray.stringify(shape2), fontsize=20)

        # set limits
        self.fig1.ax.set_ylim([ybound, -0.2*ybound])
        self.fig2.ax.set_ylim([ybound, -0.2*ybound])
        self.fig1.ax.set_xlim([0, xbound])
        self.fig2.ax.set_xlim([0, xbound])
