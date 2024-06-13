from tpxpy.loader import TpxLoader
from tpxpy.analysis import SRHomScan

import numpy as np
import matplotlib.pyplot as plt

from tkinter.filedialog import askdirectory

superres = 1

tot_calibration = "E:/Projects/SpectralHOM/calibration/tot_calibration.txt"
wl_calibration = "E:/Projects/SpectralHOM/calibration/2023-11-08 spectrum/argon.calib.csv"

dirname = askdirectory(initialdir="E:/Projects/SpectralHOM/Timepix/", mustexist=True)

tpxl = TpxLoader(compress_cache=True)
tpxl.set_tot_calibration(tot_calibration)

print(dirname)
tpxl.cache_all_files(dirname, use_existing_cache=True)

scan = SRHomScan(dirname, tpxl, superresolution=superres, calibration_file=wl_calibration)

delays, freqs_a, freqs_b, yplot_a, yplot_b = scan.yingwen_plot_singles()

freqs_a *= 1e-12
freqs_b *= 1e-12
delays *= 1e6

plt.figure()
plt.subplot(121)
plt.imshow(yplot_a, interpolation='nearest', origin='lower', aspect='auto', extent=(freqs_a[0], freqs_a[-1], delays[0], delays[-1]))
plt.xlabel('Relative Frequency [THz]')
plt.ylabel('Delay [um]')
plt.subplot(122)
plt.imshow(yplot_b, interpolation='nearest', origin='lower', aspect='auto', extent=(freqs_b[0], freqs_b[-1], delays[0], delays[-1]))
plt.xlabel('Frequency [THz]')
plt.suptitle(os.path.basename(dirname))
plt.show()
