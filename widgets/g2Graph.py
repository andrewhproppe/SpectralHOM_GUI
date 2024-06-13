import seaborn as sns

from fig_utils import dress_fig
from widgets.utils import ScalarFormatterClass
from widgets.PlotCanvas import PlotCanvas


class g2Graph(PlotCanvas):
    def __init__(self, parent=None, nchannels=4, width=3, height=2, dpi=150):
        super().__init__(parent, width, height, dpi)
        self.colors = sns.color_palette('icefire', nchannels)
        self.axes.set_xlabel('Time (ps)')
        self.axes.set_ylabel('$g^{(2)}(\\tau)$')
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 20))
        self.axes.yaxis.set_major_formatter(yScalarFormatter)
        dress_fig(tight=True)
