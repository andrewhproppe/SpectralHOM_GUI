from tpxpy.loader import TpxLoader
from tpxpy.analysis import BiphotonSpectrum, TwinBeam
import tpxpy.utils as tpxutils

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

img = tpxutils.load_dir_concatenated(tpxl, dirname)

img.set_coincidence_window(0, 10)

bispec = BiphotonSpectrum(TwinBeam(img, superresolution=superres))
bispec.load_calibration(wl_calibration)

f1, f2, homogram = bispec.hom_hologram()
_, _, jsi = bispec.joint_spectrum(type='frequency')

extent = (f1[0], f1[-1], f2[0], f2[-1])

plt.figure()
plt.subplot(131)
plt.imshow(jsi, origin='lower', interpolation='nearest', extent=extent, aspect=1)
plt.subplot(132)
plt.imshow(np.abs(homogram), origin='lower', interpolation='nearest', extent=extent, aspect=1)
plt.subplot(133)
plt.imshow(np.angle(homogram)*np.abs(homogram), origin='lower', interpolation='nearest', extent=extent, aspect=1, cmap='seismic')
plt.show()
