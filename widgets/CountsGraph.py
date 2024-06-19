import numpy as np
import seaborn as sns

from fig_utils import dress_fig
from widgets.utils import ScalarFormatterClass
from widgets.PlotCanvas import PlotCanvas

class CountsGraph(PlotCanvas):
    def __init__(self, parent=None, nchannels=8, width=3, height=2, dpi=150):
        super().__init__(parent, width, height, dpi)
        self.colors = sns.color_palette('tab10', nchannels)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Counts per second')

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        # yScalarFormatter.set_powerlimits((0, 0))
        self.axes.yaxis.set_major_formatter(yScalarFormatter)
        dress_fig(tight=True)

    def plot(self, t, data, channels, norm_data=None, norm_channels=None):
        # for artist in plt.gca().lines + plt.gca().collections:
        for artist in self.fig.gca().lines + self.fig.gca().collections:
            artist.remove()

        for i, d in enumerate(data):
            self.axes.plot(t, d, color=self.colors[i], label=channels[i])

        # TODO: figure out how to plot norm data

        # self.axes.set_ylim(np.min(data) * 0.99, np.max(data) * 1.01)
        # self.axes.set_xlim(np.min(t) * 0.99, np.max(t) * 1.01)
        min_data = np.min(data)
        max_data = np.max(data)
        buffer = 0.05  # 5% buffer around the data
        y_min = min_data - buffer * (max_data - min_data)
        y_max = max_data + buffer * (max_data - min_data)
        self.axes.set_ylim(y_min, y_max)

        # self.axes.autoscale('y')

        self.draw()
