from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd

# --- SETTINGS ---
filename = 'AIRS.2024.12.04.L2G.Precip_Est_IR.v1.0.4.1.G24344113024.hdf'
x = 44.43    # latitudine dorită (ex: București)
y = 26.10    # longitudine dorită

f = SD(filename, SDC.READ)

# Extrage variabilele principale
lat = f.select('Latitude')[:]
lon = f.select('Longitude')[:]

# Calcul distanță pătratică pixel-pixel (simplu)
dists = (lat - x) ** 2 + (lon - y) ** 2
idx = np.unravel_index(np.argmin(dists), dists.shape)

print(f"Cel mai apropiat grid-point de ({x}, {y}) este la ({lat[idx]}, {lon[idx]}) [i,j]={idx}")

# Listă cu numele tuturor variabilelor pe care vrei să le extragi
vars_to_extract = [
    'GridCenterLatitude', 'GridCenterLongitude', 'GridTime', 'Latitude', 'Longitude', 'Time',
    'granuleNumber', 'SurfClass', 'landFrac', 'CldFrcStd', 'PCldTopStd', 'IR_Precip_Est',
    'IR_Precip_Est_Err', 'IR_Precip_Est_QC', 'ScanIndex', 'PixelIndex'
]

# Extrage valorile pentru pixelul cel mai apropiat
vals = {}
for var in vars_to_extract:
    try:
        data = f.select(var)[:]
        # Asumăm că toate variabilele au aceeași structură ca Latitude; dacă nu, o adaptezi punctual!
        if data.shape == lat.shape:
            vals[var] = data[idx]
        else:
            # Dacă variabila nu are shape (lat), ia mijlocul sau primul pixel, sau adaptezi
            vals[var] = data[idx[0]] if data.ndim == 1 else data[idx]
    except Exception as e:
        vals[var] = None
        print(f"Variabila {var}: {e}")

print("\nValori extrase pentru punctul cel mai apropiat:")
for k, v in vals.items():
    print(f"{k:18s}: {v}")

# Creează DataFrame cu o singură linie dacă vrei export
df = pd.DataFrame([vals])
df.to_csv("pixel_valori_airs.csv", index=False)
print("\nSalvat în 'pixel_valori_airs.csv'")