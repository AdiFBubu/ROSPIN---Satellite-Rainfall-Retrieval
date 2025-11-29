import h5py

# Open the HDF5 file
file_path = "2A.GPM.GMI.GPROF2021v1.20250111-S013100-E030413.061720.V07C.HDF5"  # Replace with your file path
hdf_file = h5py.File(file_path, "r")

# Access surfacePrecipitation data
surface_precip = hdf_file['S1/surfacePrecipitation'][:]
print("Surface Precipitation Data Shape:", surface_precip.shape)

# Check associated attributes for metadata
print("\nAttributes of Surface Precipitation Dataset:")
for attr_name, attr_value in hdf_file['S1/surfacePrecipitation'].attrs.items():
    print(f"{attr_name}: {attr_value}")

# Check ScanTime to understand temporal resolution
scan_time_day = hdf_file['S1/ScanTime/DayOfMonth'][:]
scan_time_hour = hdf_file['S1/ScanTime/Hour'][:]
scan_time_min = hdf_file['S1/ScanTime/Minute'][:]

# Extrage vectorul de latitudine și longitudine
lat = hdf.select('GridCenterLatitude')[:]
try:
    lon = hdf.select('GridCenterLongitude')[:]
except:
    print("Nu există GridCenterLongitude – vezi numele corect din lista dataseturilor tale!")

print(lat.shape)
print(lat)
print(lon.shape)
print(lon)

print("\nScan Time (First 10 Observations):")
for i in range(10):
    print(f"Day: {scan_time_day[i]}, Hour: {scan_time_hour[i]}, Minute: {scan_time_min[i]}")

# Close the file
hdf_file.close()