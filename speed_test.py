import time
import requests
import planetary_computer as pc
import pystac_client

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=[76.55, 12.25, 76.70, 12.35], datetime="2026-08-18/2026-09-17", limit=1).items())[0]
href = pc.sign(s1.assets["vv"].href)

t0 = time.time()
r = requests.get(href, stream=True)
chunk_size = 1024 * 1024  # 1 MB
downloaded = 0
for chunk in r.iter_content(chunk_size=chunk_size):
    downloaded += len(chunk)
    if downloaded >= 10 * 1024 * 1024:  # 10 MB
        break
t1 = time.time()
speed = (downloaded / (1024 * 1024)) / (t1 - t0)
print(f"Downloaded {downloaded/(1024*1024):.1f} MB in {t1-t0:.2f}s ({speed:.2f} MB/s)")
