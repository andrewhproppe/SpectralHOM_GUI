from fig_utils import *
from tkinter import filedialog

matplotlib.use('Qt5Agg')
set_font_size(8)

root = 'G:\Shared drives\JCEP Lab\Projects\Spectral HOM\HOM stage scan'
fnames = filedialog.askopenfilenames(initialdir=root)

shift_min = False
normalize = True

stage_pos = []
g2 = []
for i, fname in enumerate(fnames):
    data = np.genfromtxt(fname, delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    if shift_min:
        min_idx = np.where(y == y.min())[0][0]
        x -= x[min_idx]
    if normalize:
        y /= y.max()
    stage_pos.append(x)
    g2.append(y)

plt.figure()
for i in range(len(g2)):
    # plt.scatter(stage_pos[i], g2[i], marker='s', s=2, linewidths=1)
    plt.plot(stage_pos[i], g2[i], '-s', markersize=2)
# plt.plot(stage_pos, g2)
plt.show()