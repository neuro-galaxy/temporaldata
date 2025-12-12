from temporaldata.io import load_nwb
import h5py
import remfile
from pynwb import NWBHDF5IO

s3_url = "https://api.dandiarchive.org/api/assets/2b9e441b-56bc-4be2-893e-0e02d22d239d/download/"

cache_dirname = '/tmp/remfile_cache'
disk_cache = remfile.DiskCache(cache_dirname)

rem_file = remfile.File(s3_url, disk_cache=disk_cache)

h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

data = load_nwb(nwbfile, lazy_loading=True)

print(data)