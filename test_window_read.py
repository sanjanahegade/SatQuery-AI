import time
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import pystac_client
import planetary_computer as pc

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]

print("Testing Sentinel-2 window reading...")
t0 = time.time()
s2 = list(client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2026-08-18/2026-09-17", sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}], limit=1).items())[0]
b04_url = pc.sign(s2.assets["B04"].href)

with rasterio.open(b04_url) as src:
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox)
    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    # round window to integer pixels
    window = window.round_offsets().round_lengths()
    print("S2 UTM bounds:", minx, miny, maxx, maxy)
    print("S2 Window:", window)
    data = src.read(1, window=window)
    print(f"S2 B04 read shape: {data.shape}, min: {data.min()}, max: {data.max()}, time: {time.time()-t0:.2f}s")
    s2_tags = src.tags()
    s2_meta = src.meta.copy()

print("\nTesting Sentinel-1 window reading...")
t1 = time.time()
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=bbox, datetime="2026-08-18/2026-09-17", limit=1).items())[0]
vv_url = pc.sign(s1.assets["vv"].href)

with rasterio.open(vv_url) as src:
    print("S1 crs:", src.crs)
    print("S1 transform:", src.transform)
    print("S1 tags:", src.tags())
    if src.crs is None:
        # Check GCPs or RPCs or use STAC proj:transform / EPSG:4326
        from rasterio.transform import Affine
        proj_trans = s1.properties.get("proj:transform")
        if proj_trans:
            trans = Affine(*proj_trans[:6])
            window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], transform=trans)
        else:
            window = None
    else:
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox) if src.crs != "EPSG:4326" else bbox
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    
    if window:
        window = window.round_offsets().round_lengths()
        print("S1 Window:", window)
        data_s1 = src.read(1, window=window)
        print(f"S1 VV read shape: {data_s1.shape}, min: {data_s1.min()}, max: {data_s1.max()}, time: {time.time()-t1:.2f}s")
    else:
        print("Could not compute S1 window from bounds directly.")
