import requests
import planetary_computer as pc
import pystac_client

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=[76.55, 12.25, 76.70, 12.35], datetime="2026-08-18/2026-09-17", limit=1).items())[0]
href = pc.sign(s1.assets["vv"].href)

r = requests.get(href, headers={"Range": "bytes=0-1023"})
print("Status code:", r.status_code)
print("Content-Range:", r.headers.get("Content-Range"))
print("Bytes received:", len(r.content))
