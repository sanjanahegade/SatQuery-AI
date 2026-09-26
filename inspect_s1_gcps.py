import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pystac_client
import planetary_computer as pc

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=bbox, datetime="2026-08-18/2026-09-17", limit=1).items())[0]
vv_url = pc.sign(s1.assets["vv"].href)

with rasterio.open(vv_url) as src:
    print("src.shape:", src.shape)
    print("src.crs:", src.crs)
    gcps, gcp_crs = src.gcps
    print(f"src.gcps: count={len(gcps)}, crs={gcp_crs}")
    if gcps:
        for g in gcps[:5]:
            print(f"  GCP: id={g.id}, row={g.row}, col={g.col}, x(lon)={g.x}, y(lat)={g.y}, z={g.z}")
        # Find row, col range for bbox [76.55, 12.25, 76.70, 12.35]
        matching_rows = [g.row for g in gcps if 76.0 <= g.x <= 77.2 and 11.8 <= g.y <= 12.8]
        matching_cols = [g.col for g in gcps if 76.0 <= g.x <= 77.2 and 11.8 <= g.y <= 12.8]
        print("Matching GCP rows range:", min(matching_rows, default=None), max(matching_rows, default=None))
        print("Matching GCP cols range:", min(matching_cols, default=None), max(matching_cols, default=None))
