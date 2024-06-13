from tpxpy.loader import TpxLoader
from tkinter.filedialog import askdirectory

tot_calibration = "E:/Projects/SpectralHOM/calibration/tot_calibration.txt"

dirname = askdirectory(mustexist=True)
print(dirname)

tpxl = TpxLoader(compress_cache=True)
tpxl.set_tot_calibration(tot_calibration)

tpxl.cache_all_files(dirname, use_existing_cache=True)

