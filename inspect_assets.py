import pystac_client
import planetary_computer as pc
import rasterio
import requests

endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
client = pystac_client.Client.open(endpoint, modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]
datetime_range = "2026-08-18/2026-09-17"

s2 = list(client.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=datetime_range,
    sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    limit=1
).items())[0]

print("S2 item:", s2.id)
print("S2 properties:", s2.properties.keys())
for b in ["B04", "B03", "B02"]:
    asset = s2.assets[b]
    signed = pc.sign(asset.href)
    print(f"  {b}: media_type={asset.media_type}")
    with rasterio.open(signed) as src:
        print(f"    src: shape=({src.count}, {src.height}, {src.width}), dtype={src.dtypes[0]}, crs={src.crs}, desc={src.descriptions}, tags={src.tags()}")

s1 = list(client.search(
    collections=["sentinel-1-grd"],
    bbox=bbox,
    datetime=datetime_range,
    limit=1
).items())[0]

print("\nS1 item:", s1.id)
print("S1 properties:", s1.properties)
vv_asset = s1.assets["vv"]
signed_vv = pc.sign(vv_asset.href)
print(f"  vv: media_type={vv_asset.media_type}")
with rasterio.open(signed_vv) as src:
    print(f"    src: shape=({src.count}, {src.height}, {src.width}), dtype={src.dtypes[0]}, crs={src.crs}, desc={src.descriptions}, tags={src.tags()}")
