import re
import numpy as np
from pyhdf.SD import SD, SDC
from pyproj import Proj, Transformer

# Constante grilă MODIS Sinusoidal
R = 6371007.181  # rază sferă (m)
NTILES_H = 36
NTILES_V = 18
TILE_SIZE = 1200  # pixeli per tile (1 km)
TILE_WIDTH = 2 * np.pi * R / NTILES_H   # ~ 1,111,950.5198 m
TILE_HEIGHT = np.pi * R / NTILES_V      # ~ 1,111,950.5198 m
PIXEL_SIZE = TILE_WIDTH / TILE_SIZE     # ~ 926.6254331 m

# Originea globală a grilei sinusoidale (colțul stânga-sus al grilei mondiale)
GLOBAL_UL_X = -np.pi * R                # ~ -20,015,109.354 m
GLOBAL_UL_Y =  np.pi * R / 2            # ~ 10,007,554.677 m

# Proiecția sinusoidală MODIS
proj_sinu = Proj("+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs")
transformer = Transformer.from_proj("epsg:4326", proj_sinu, always_xy=True)

def parse_tile_from_filename(path):
    m = re.search(r"h(\d{2})v(\d{2})", path)
    if not m:
        raise ValueError("Nu pot găsi patternul hXXvYY în numele fișierului.")
    return int(m.group(1)), int(m.group(2))

def lonlat_to_tile_xy(lon, lat, h, v):
    # Coord. metrice sinusoidale pentru punctul țintă
    x, y = transformer.transform(lon, lat)
    # Colțul stânga-sus al tile-ului (h, v)
    tile_ul_x = GLOBAL_UL_X + h * TILE_WIDTH
    tile_ul_y = GLOBAL_UL_Y - v * TILE_HEIGHT
    # Index pixel în tile
    ix = int(np.floor((x - tile_ul_x) / PIXEL_SIZE))
    iy = int(np.floor((tile_ul_y - y) / PIXEL_SIZE))
    return ix, iy, x, y, tile_ul_x, tile_ul_y

def read_at(hdf, var, iy, ix):
    ds = hdf.select(var)
    dims = ds.dimensions()
    rank = len(dims)
    if rank != 2:
        raise ValueError(f"{var} are rank {rank}, mă așteptam 2D (tile 1200x1200). Dims: {dims}")
    val = ds.get(start=(int(iy), int(ix)), count=(1, 1)).squeeze()

    # scale/offset/fill
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
    return float(val)

def decode_qc_day(qc_value):
    # QC_Day este bitfield pe 8 biți (documentația MOD11A1 v6/v61)
    # Bits 0-1: Mandatory QA (0=good, 1=other, 2=cloud, 3=not produced)
    # Bits 2-3: Data quality (0=good, 1=other, 2=not defined, 3=not defined)
    # Bits 4-5: Emissivity error (0:<0.01, 1:0.01–0.02, 2:0.02–0.03, 3:>=0.03)
    # Bits 6-7: LST error (0:<1K, 1:1–2K, 2:2–3K, 3:>=3K)
    if np.isnan(qc_value):
        return {"qc_raw": np.nan}
    q = int(qc_value) & 0xFF
    return {
        "qc_raw": q,
        "mandatory_qa": (q >> 0) & 0b11,
        "data_quality":  (q >> 2) & 0b11,
        "emis_error":    (q >> 4) & 0b11,
        "lst_error":     (q >> 6) & 0b11,
    }

def main():
    hdf_path = "MOD11A1.A2025316.h10v05.061.2025317123700.hdf"
    hdf = SD(hdf_path, SDC.READ)

    # Afișează dimensiuni pentru QC_Day (cum ai verificat)
    ds = hdf.select('QC_Day')
    dims = ds.dimensions()
    print(f"Dimensiuni: {dims}")
    print(f"Shape: {tuple(dims.values())}")

    # Coordonate țintă
    lat_target = 30.0
    lon_target = -90.0

    # Tile din nume
    h, v = parse_tile_from_filename(hdf_path)

    # Calculează indexul pixelului în tile
    ix, iy, x, y, tile_ul_x, tile_ul_y = lonlat_to_tile_xy(lon_target, lat_target, h, v)

    # Verifică bounds
    if not (0 <= ix < TILE_SIZE and 0 <= iy < TILE_SIZE):
        print(f"Punctul (lat={lat_target}, lon={lon_target}) NU este în tile-ul h{h:02d}v{v:02d}.")
        print(f"Indice calculat: ix={ix}, iy={iy} (valid 0..{TILE_SIZE-1})")
        return

    print(f"Pixel în tile h{h:02d}v{v:02d}: iy={iy}, ix={ix}")
    print(f"Coord. sinu x,y (m): {x:.2f}, {y:.2f}; Tile UL x,y: {tile_ul_x:.2f}, {tile_ul_y:.2f}")

    var_names = [
        'LST_Day_1km',
        'LST_Night_1km',
        'Emis_31',
        'Emis_32',
    ]

    results = {}
    for var in var_names:
        try:
            val = read_at(hdf, var, iy, ix)
        except Exception as e:
            val = f"Eroare: {e}"
        results[var] = val

    # Decodare QC_Day/QC_Night dacă sunt numerice
    qc_day_info = decode_qc_day(results.get('QC_Day') if isinstance(results.get('QC_Day'), (int, float)) else np.nan)
    qc_night_info = decode_qc_day(results.get('QC_Night') if isinstance(results.get('QC_Night'), (int, float)) else np.nan)

    # Afișare
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()