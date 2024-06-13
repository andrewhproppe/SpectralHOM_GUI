from tpxpy.loader import TpxLoader
from tpxpy.analysis import SRHomScan

import numpy as np
import matplotlib.pyplot as plt

from tkinter.filedialog import askdirectory

superres = 3

tot_calibration = "E:/Projects/SpectralHOM/calibration/tot_calibration.txt"
wl_calibration = "E:/Projects/SpectralHOM/calibration/2023-11-08 spectrum/argon.calib.csv"

dirname = askdirectory(initialdir="E:/Projects/SpectralHOM/Timepix/", mustexist=True)

tpxl = TpxLoader(compress_cache=True)
tpxl.set_tot_calibration(tot_calibration)

print(dirname)
tpxl.cache_all_files(dirname, use_existing_cache=True)

scan = SRHomScan(dirname, tpxl, superresolution=superres, calibration_file=wl_calibration)

delays, freqs, yplot = scan.yingwen_plot()
amp, phase, offset, d_amp, d_phase, d_offset = scan.fit_yingwen_plot()

plt.figure()
plt.subplot(121)
plt.imshow(yplot, interpolation='nearest', origin='lower', aspect='auto', extent=(freqs[0], freqs[-1], delays[0], delays[-1]))
plt.xlabel('Relative Frequency [Hz]')
plt.ylabel('Delay [m]')
ax = plt.subplot(122)
ax.plot(freqs, amp/np.max(amp), 'b-')
ax2 = ax.twinx()
ax2.plot(freqs, np.unwrap(phase), 'r:')
plt.suptitle(os.path.basename(dirname))
plt.show()