import numpy as np
import seaborn as sns

from fig_utils import dress_fig
from widgets.utils import ScalarFormatterClass
from widgets.PlotCanvas import PlotCanvas


class HOMGraph(PlotCanvas):
    def __init__(self, parent=None, nchannels=4, width=3, height=2, dpi=150):
        super().__init__(parent, width, height, dpi)
        self.colors = sns.color_palette('icefire', nchannels)
        self.axes.set_xlabel('Stage position (mm)')
        self.axes.set_ylabel('$g^{(2)}(\\tau)$')
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 20))
        self.axes.yaxis.set_major_formatter(yScalarFormatter)
        dress_fig(tight=True)

    def plot(self, positions, data):
        for artist in self.fig.gca().lines + self.fig.gca().collections:
            artist.remove()

        self.axes.plot(positions, data, color=self.colors[0])
        self.axes.autoscale(axis='y')
        # self.axes.set_ylim(np.min(data) * 0.99, np.max(data) * 1.01)
        step = np.diff(positions)[0]
        self.axes.set_xlim(np.min(positions) - step, np.max(positions) + step)
        # self.axes.set_xlim(np.min(positions), np.max(positions))
        # self.axes.autoscale('y')
        self.draw()
