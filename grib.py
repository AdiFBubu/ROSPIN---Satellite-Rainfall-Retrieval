import pandas as pd
import numpy as np

import xarray as xr

ds = xr.open_dataset("80207865bf766aaba4752a2f1af7dccc.grib", engine="cfgrib")
# Varianta 1: Listezi toate variabilele de date (data variables)
print(ds.data_vars)

# Varianta 2: Listezi doar numele variabilelor
print(list(ds.data_vars))
lat_req = 30.0
lon_req = -90.0

# Indexul cel mai apropiat pentru lat și lon
lat_idx = np.argmin(np.abs(ds.latitude.values - lat_req))
lon_idx = np.argmin(np.abs(ds.longitude.values - lon_req))

# extrage valorile și valid_time pentru fiecare pas și timp
vals = ds['tcc'][:, lat_idx, lon_idx].values           # (5115, 12)
valid_times = ds['time'][:].values                  # (5115, 12)
# # reshape la un vector, ca să fie câte un rând pentru fiecare predicție punctuală
# vals = vals.ravel()
# valid_times = valid_times.ravel()
#
df = pd.DataFrame({
    "Time": valid_times,
    "avg_vimdf": vals
})

df.to_csv("tcc.csv", index=False)


# print(ds['avg_ie'])

# Construiește DataFrame
# df = pd.DataFrame({
#     'Time': datetimes,
#     'avg_ie': vals,
# })
#
# df.to_csv("mer.csv", index=False)