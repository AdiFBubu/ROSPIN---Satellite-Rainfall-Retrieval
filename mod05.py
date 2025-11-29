import numpy as np
from pyhdf.SD import SD, SDC

hdf = SD('MOD05_L2.A2024043.1550.061.2024044020616.hdf', SDC.READ)

# Find nearest grid point
lat = hdf.select('Latitude')[:]
lon = hdf.select('Longitude')[:]
lat_target = 30
lon_target = -90

dist = (lat - lat_target)**2 + (lon - lon_target)**2
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Cel mai apropiat grid point: iy={iy}, ix={ix} (lat={lat[iy,ix]}, lon={lon[iy,ix]})")

var_names = [
    'Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth',
    'Cloud_Mask_QA', 'Water_Vapor_Near_Infrared', 'Water_Vapor_Correction_Factors',
    'Water_Vapor_Infrared', 'Quality_Assurance_Near_Infrared', 'Quality_Assurance_Infrared'
]

def read_point(ds, iy, ix):
    # Determine rank from dimensions()
    dims = ds.dimensions()  # dict: {dim_name: size} in order
    dim_names = list(dims.keys())
    dim_sizes = list(dims.values())
    rank = len(dim_sizes)

    # Some SDS may have empty dims; fall back to info()
    if rank == 0:
        info = ds.info()
        # info[1] can be a tuple of sizes
        if isinstance(info[1], (list, tuple)):
            dim_sizes = list(info[1])
            rank = len(dim_sizes)
        elif isinstance(info[1], int):
            dim_sizes = [info[1]]
            rank = 1

    # Build start/count of proper length
    if rank == 2:
        start = (int(iy), int(ix))
        count = (1, 1)
    elif rank == 3:
        # Assume third dimension is band/level; take first element
        start = (int(iy), int(ix), 0)
        count = (1, 1, 1)
    elif rank == 1:
        # Choose index along the single dimension.
        # If dim name looks like 'XDim' or has 'Columns', use ix; otherwise iy.
        dn = dim_names[0] if dim_names else ''
        if any(k in dn.lower() for k in ['x', 'col', 'lon']):
            idx = ix
        else:
            idx = iy
        # Clamp to valid range
        idx = max(0, min(int(idx), dim_sizes[0] - 1))
        start = (idx,)
        count = (1,)
    else:
        raise ValueError(f"Unsupported rank {rank} with dims {dim_names} sizes {dim_sizes}")

    arr = ds.get(start=start, count=count)
    # Squeeze scalar
    val = np.array(arr).squeeze()

    # Apply scale/offset if present
    attrs = ds.attributes()
    scale = attrs.get('scale_factor') or attrs.get('Scale') or attrs.get('Scale_Factor')
    offset = attrs.get('add_offset') or attrs.get('Offset') or attrs.get('Add_Offset')
    fill = attrs.get('_FillValue') or attrs.get('fill_value') or attrs.get('FillValue')

    if fill is not None and np.equal(val, fill).all():
        return np.nan
    if scale is not None:
        val = val * scale
    if offset is not None:
        val = val + offset

    return val

results = {}
for var in var_names:
    try:
        ds = hdf.select(var)
    except Exception as e:
        results[var] = f"Nu pot deschide SDS '{var}': {e}"
        continue

    try:
        val = read_point(ds, iy, ix)
    except Exception as e:
        # Log diagnostics to help understand shape/rank
        try:
            dims = ds.dimensions()
            dim_names = list(dims.keys())
            dim_sizes = list(dims.values())
        except Exception:
            dim_names, dim_sizes = [], []
        results[var] = f"Eroare la citire: {e} (dims: {dim_names} sizes: {dim_sizes})"
        continue

    results[var] = val

for var, val in results.items():
    print(f"{var}: {val}")