from numpy import asarray, hstack, arange
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import xlim, ylim, gca, axis, figure, text


class DisplayArray(object):

    def __init__(self, width=600, height=200):
        self.width = width
        self.height = height

        fig = figure(figsize=(12, 6))
        self.fig = fig
        self.ax = gca()

    def horzbox(self, x, y, width, scale, color):
        rect = Rectangle((x, y), width, scale, fill=True, color=color, ec='white', lw=5)
        self.ax.add_patch(rect)
        self.updatebounds(x + width, y + scale)

    def vertbox(self, x, y, height, scale, color):
        rect = Rectangle((x, y), scale, height, fill=True, color=color, ec='white', lw=5)
        self.ax.add_patch(rect)
        self.updatebounds(x + scale, y + height)

    def updatebounds(self, x, y):
        self.xmax = max(self.xmax, x)
        self.ymax = max(self.ymax, y)

    @staticmethod
    def strformat(shape):
        return (("%s x " * len(shape)) % (shape))[:-3]

    @staticmethod
    def iseven(n):
        if (n == 0) | (n % 2 == 0):
            return 0
        else:
            return 1

    def draw(self, shape1, shape2):

        self.init(shape1, shape2)

        left = 0
        above = 0

        if len(shape1) > 0:
            text(0, -20, self.strformat(shape1), fontsize=20)
        elif len(shape2) > 0:
            text(0, -20, self.strformat(shape2), fontsize=20)

        for i, s in enumerate(self.shape):
            shift = s * self.scale
            if i >= len(shape1):
                clr = self.cmap1(self.count[i]/float(len(shape2) + 0.1))
            else:
                clr = self.cmap2(self.count[i]/float(len(shape1) + 0.1))
            if self.pos[i] == 0:
                self.horzbox(left, above, shift, self.scale, clr)
                left += shift
            else:
                self.vertbox(left, above, shift, self.scale, clr)
                above += shift

            if i < (len(self.shape) - 1):
                if self.count[i+1] == 0:
                    if self.pos[i] == 0:
                        left += self.scale
                    else:
                        left += self.scale * 2
                    above = 0
                    text(left, -20, self.strformat(shape2), fontsize=20)

        self.ax.set_aspect('equal')
        xlim([0, self.xmax])
        ylim([self.ymax, -40])

    def init(self, shape1, shape2):

        shape1 = asarray(shape1, 'float')
        shape2 = asarray(shape2, 'float')

        n = len(shape1)
        m = len(shape2)

        rescale = min(hstack((shape1, shape2)))
        shape1 = shape1 / rescale
        shape2 = shape2 / rescale

        pos1 = asarray(map(lambda x: self.iseven(x), range(n)))
        pos2 = asarray(map(lambda x: self.iseven(x), range(m)))

        depth1 = sum(shape1[pos1 == 1])
        depth2 = sum(shape2[pos2 == 1])

        shape = hstack((shape1, shape2))
        pos = hstack((pos1, pos2))
        count = hstack((arange(n), arange(m)))

        ncol = sum(shape[pos == 0])
        nrow = max(depth1, depth2)

        if len(shape1) > 0:
            if pos1[-1] == 0:
                ncol += 1
            if pos1[-1] == 1:
                ncol += 2
            if pos1[-1] == 1:
                ncol += 1

        if len(shape2) > 0:
            if pos2[-1] == 1:
                ncol += 1

        if len(shape1) > 0 and len(shape2) > 0:
            if depth1 > depth2:
                if pos1[-1] == 0:
                    nrow += 1
            else:
                if pos2[-1] == 0:
                    nrow += 1

        if ncol > nrow:
            scale = min([self.height / nrow, self.width / ncol])
        else:
            scale = self.height / nrow

        self.shape = shape
        self.pos = pos
        self.count = count
        self.scale = scale
        self.xmax = 0
        self.ymax = 0
        self.cmap1 = LinearSegmentedColormap.from_list('blend', ["#393B79", "#9C9EDE"])
        self.cmap2 = LinearSegmentedColormap.from_list('blend', ["#E6550D", "#FDD0A2"])

