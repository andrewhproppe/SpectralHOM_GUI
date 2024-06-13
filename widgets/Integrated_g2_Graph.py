import numpy
import seaborn as sns

from fig_utils import dress_fig
from widgets.utils import ScalarFormatterClass
from widgets.PlotCanvas import PlotCanvas


class Integrated_g2_Graph(PlotCanvas):
    def __init__(self, parent=None, nchannels=8, width=3, height=2, dpi=150):
        super().__init__(parent, width, height, dpi)
        self.colors = sns.color_palette('tab10', nchannels)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Integrated g2(t)')

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        # yScalarFormatter.set_powerlimits((0, 0))
        self.axes.yaxis.set_major_formatter(yScalarFormatter)
        dress_fig(tight=True)

    def plot(self, t, data):
        # for artist in plt.gca().lines + plt.gca().collections:
        for artist in self.fig.gca().lines + self.fig.gca().collections:
            artist.remove()

        # for i, d in enumerate(data):
        self.axes.plot(t, data, color=self.colors[0])

        min_data = np.min(data)
        max_data = np.max(data)
        buffer = 0.05  # 5% buffer around the data
        y_min = min_data - buffer * (max_data - min_data)
        y_max = max_data + buffer * (max_data - min_data)
        self.axes.set_ylim(y_min, y_max)

        min_t = np.min(t)
        max_t = np.max(t)
        buffer = 0.05  # 5% buffer around the data
        x_min = min_t - buffer * (max_t - min_t)
        x_max = max_t + buffer * (max_t - min_t)
        self.axes.set_xlim(x_min, x_max)

        self.draw()
