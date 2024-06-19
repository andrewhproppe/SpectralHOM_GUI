import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from fig_utils import *
from tkinter import filedialog
from scipy.optimize import curve_fit
matplotlib.use('Qt5Agg')
set_font_size(8)

# Define the Gaussian function
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * (wid/2.355)**2))

def fit_function(x, amp1, cen1, wid1, amp2, cen2):
    return gaussian(x, amp1, cen1, wid1) + gaussian(x, amp2, cen2, wid1)

root = 'G:\Shared drives\JCEP Lab\Projects\Spectral HOM\g2s'
fnames = filedialog.askopenfilenames(initialdir=root)
colors = sns.color_palette("tab10", 6)
# labels = ['signal path', 'idler path']
labels = ['1', '2', '3', '4', '5', '6']
distances = [0, 5, 10, 15, 20, 25]
fit_peaks = False
initial_guess = [1, -176, 10, 1, 114]
peak_diffs = []

plt.figure()
t_max = []
for i, fname in enumerate(fnames):
    data = np.genfromtxt(fname, delimiter=',')
    t  = data[1:, 0]
    g2 = data[1:, 1]
    g2 = g2/np.max(g2)
    if fit_peaks:
        fitted_params, _ = curve_fit(fit_function, t, g2, p0=initial_guess)
        fit = fit_function(t, *fitted_params)
        peak_diff = fitted_params[4] - fitted_params[1]
        peak_diffs.append(peak_diff)
        plt.plot(t, fit, color=colors[i])
    t_max.append(t[np.where(g2==1)])
    label = fname.split('/g2s/')[-1]
    plt.plot(
        t,
        g2,
        # label=labels[i],
        label=label,
        color=colors[i]
    )

plt.legend()
plt.tight_layout()
# plt.figure(figsize=(4, 2.5), dpi=150)
# plt.plot(steps, ch1/chsum, marker='s', ms=3, label='ch1')
# plt.plot(steps, ch2/chsum, marker='s', ms=3, label='ch2')
# # plt.plot(steps, (ch1-ch2)/(chsum), marker='s', ms=3, label='ch diff')
# plt.xlabel('Step (mm)')
# plt.legend()
# plt.tight_layout()
# plt.show()