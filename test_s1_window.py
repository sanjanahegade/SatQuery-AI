import rasterio
from rasterio.warp import transform_geom
from rasterio.features import geometry_mask
import pystac_client
import planetary_computer as pc
import numpy as np

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=bbox, datetime="2026-08-18/2026-09-17", limit=1).items())[0]
vv_url = pc.sign(s1.assets["vv"].href)

with rasterio.open(vv_url) as src:
    gcps, gcp_crs = src.gcps
    print("S1 GCPs:", len(gcps))
    # We can fit a polynomial or bilinear interpolation from lon/lat to row/col
    # or use scipy.interpolate.griddata
    from scipy.interpolate import LinearNDInterpolator
    lons = [g.x for g in gcps]
    lats = [g.y for g in gcps]
    rows = [g.row for g in gcps]
    cols = [g.col for g in gcps]
    interp_r = LinearNDInterpolator(list(zip(lons, lats)), rows)
    interp_c = LinearNDInterpolator(list(zip(lons, lats)), cols)
    
    # Bbox corners: (76.55, 12.25), (76.70, 12.25), (76.70, 12.35), (76.55, 12.35)
    corners_lon = [76.55, 76.70, 76.70, 76.55]
    corners_lat = [12.25, 12.25, 12.35, 12.35]
    r_pts = interp_r(corners_lon, corners_lat)
    c_pts = interp_c(corners_lon, corners_lat)
    print("Corner rows:", r_pts)
    print("Corner cols:", c_pts)
    
    row_min = int(max(0, np.floor(np.min(r_pts))))
    row_max = int(min(src.height, np.ceil(np.max(r_pts))))
    col_min = int(max(0, np.floor(np.min(c_pts))))
    col_max = int(min(src.width, np.ceil(np.max(c_pts))))
    
    print(f"Computed pixel window: rows=[{row_min}, {row_max}], cols=[{col_min}, {col_max}]")
    window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
    print("Window:", window)
    data = src.read(1, window=window)
    print(f"Read shape: {data.shape}, min={data.min()}, max={data.max()}")
    tags = src.tags()
    print("Tags preserved from source:", tags)
