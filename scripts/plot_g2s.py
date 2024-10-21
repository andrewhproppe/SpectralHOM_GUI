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
#
# def fit_function(x, amp1, cen1, wid1, amp2, cen2):
#     return gaussian(x, amp1, cen1, wid1) + gaussian(x, amp1, cen2, wid1)
#     # return gaussian(x, amp1, cen1, wid1) + gaussian(x, amp2, cen2, wid1)

def fit_function(x, amp1, cen1, wid1):
    return gaussian(x, amp1, cen1, wid1)
    # return gaussian(x, amp1, cen1, wid1) + gaussian(x, amp2, cen2, wid1)

root = r'G:\Shared drives\JCEP Lab\Projects\Spectral HOM\g2s'
fnames = filedialog.askopenfilenames(initialdir=root)
colors = sns.color_palette("tab10", 10)
# labels = ['signal path', 'idler path']
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',]
# distances = [0, 5, 10, 15, 20, 25]
# distances = [26, 23, 20, 17, 14, 11, 8, 5, 2]
distances = [2, 5, 8, 11, 14, 17, 20, 23, 26]
fit_peaks = False
spacer = 0.
                #amp1, cen1, wid1, amp2, cen2
# initial_guess = [   1,    0,   100,    0,   0]
initial_guess = [   1,    0,   100]
peak_diffs = []
peak_widths =[]

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
        # peak_diff = fitted_params[4] - fitted_params[1]
        # peak_diffs.append(peak_diff)
        peak_widths.append(fitted_params[-1])
        plt.plot(t, fit + i*spacer, color=colors[i])
    t_max.append(t[np.where(g2==1)])
    label = fname.split('/g2s/')[-1]
    plt.plot(
        t,
        g2 + i*spacer,
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

x_points = [25, 20, 15, 10, 5, 0]
y_points = [388, 381, 399, 431, 490, 565]