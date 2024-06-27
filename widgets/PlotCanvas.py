from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=150):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, t, data):
        # for artist in plt.gca().lines + plt.gca().collections:
        for artist in self.fig.gca().lines + self.fig.gca().collections:
            artist.remove()

        for i, d in enumerate(data):
            self.axes.plot(t, d, color=self.colors[i])

        self.axes.set_ylim(np.min(data) * 0.99, np.max(data) * 1.01)
        self.axes.set_xlim(np.min(t) * 0.99, np.max(t) * 1.01)
        # self.axes.autoscale('y')
        self.draw()
