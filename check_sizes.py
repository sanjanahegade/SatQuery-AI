import pystac_client
import planetary_computer as pc
import requests

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [76.55, 12.25, 76.70, 12.35]
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=bbox, datetime="2026-08-18/2026-09-17", limit=1).items())[0]
vv_href = pc.sign(s1.assets["vv"].href)
r = requests.head(vv_href)
content_len = int(r.headers.get("Content-Length", 0))
print(f"S1 VV asset Content-Length: {content_len / (1024*1024):.2f} MB")

s2 = list(client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2026-08-18/2026-09-17", sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}], limit=1).items())[0]
for b in ["B04", "B03", "B02"]:
    b_href = pc.sign(s2.assets[b].href)
    r2 = requests.head(b_href)
    print(f"S2 {b} Content-Length: {int(r2.headers.get('Content-Length', 0)) / (1024*1024):.2f} MB")
