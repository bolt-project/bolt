from numpy import asarray, hstack, arange
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import xlim, ylim, gca, figure, text, subplot


class DisplayArray(object):

    def __init__(self, width=600, height=200, ax=None):
        self.width = width
        self.height = height

        if ax is None:
            self.fig = figure(figsize=(12, 6))
            self.ax = gca()

        else:
            self.ax = ax

    def box(self, x, y, width, height, color):
        rect = Rectangle((x, y), width, height, fill=True, color=color, ec='white', lw=5)
        self.ax.add_patch(rect)

    @staticmethod
    def strformat(shape):
        return (("%s x " * len(shape)) % shape)[:-3]

    @staticmethod
    def iseven(n):
        if (n == 0) | (n % 2 == 0):
            return 0
        else:
            return 1

    def draw(self, shape, cmap=None):

        if cmap is None:
            cmap = LinearSegmentedColormap.from_list('blend', ["#E6550D", "#FDD0A2"])

        sizes, orient, scale = self.init(shape)
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

        text(0, -20, self.strformat(shape), fontsize=20)

        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_xlim([0, xmax])
        self.ax.set_ylim([ymax, -40])
        self.xmax = xmax
        self.ymax = ymax

        return self

    def init(self, shape):

        shape = asarray(shape, 'float')
        n = len(shape)

        # normalize dims to 1
        shape /= shape.min()

        # set if boxes will be horzizontal vs vertical
        orient = asarray(map(lambda x: self.iseven(x), range(n)))
        ncol = sum(shape[orient == 0])
        nrow = sum(shape[orient == 1])

        if orient[-1] == 0:
            nrow += 1
        if orient[-1] == 1:
            ncol += 2

        if ncol > nrow:
            scale = min([self.height / nrow, self.width / ncol])
        else:
            scale = self.height / nrow

        return shape, orient, scale

class DisplayArrayJoint(object):

    def __init__(self):
        self.fig = figure(figsize=(10, 5))

    def draw(self, shape1, shape2):

        cmap = LinearSegmentedColormap.from_list('blend', ["#393B79", "#9C9EDE"])

        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax0 = subplot(gs[0])
        fig1 = DisplayArray(ax=ax0).draw(shape1)
        ax1 = subplot(gs[1])
        fig2 = DisplayArray(ax=ax1).draw(shape2, cmap=cmap)

        ybound = max(fig1.ymax, fig2.ymax)
        fig1.ax.set_ylim([ybound, -40])
        fig2.ax.set_ylim([ybound, -40])

        gs.set_width_ratios(([fig1.xmax / fig2.xmax, 1]))

        self.fig.tight_layout()