from tpxpy.loader import TpxLoader
from tpxpy.analysis import BiphotonSpectrum, TwinBeam
import tpxpy.utils as tpxutils

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

data = tpxutils.load_dir_concatenated(tpxl, dirname)
data.set_coincidence_window(0, 10)
bispec = BiphotonSpectrum(TwinBeam(data, superresolution=superres))
bispec.load_calibration(wl_calibration)

f1, f2, jsi = bispec.joint_spectrum(type='frequency')

plt.figure()
plt.imshow(tpxutils.orient(jsi), origin='lower', aspect='auto', extent=(f1[0], f1[-1], f2[0], f2[-1]), interpolation='nearest')
plt.show()
