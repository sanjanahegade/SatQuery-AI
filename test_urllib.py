import time
import urllib.request
import planetary_computer as pc
import pystac_client

client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
s1 = list(client.search(collections=["sentinel-1-grd"], bbox=[76.55, 12.25, 76.70, 12.35], datetime="2026-08-18/2026-09-17", limit=1).items())[0]
href = pc.sign(s1.assets["vv"].href)

t0 = time.time()
req = urllib.request.Request(href, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    data = resp.read(15 * 1024 * 1024)
t1 = time.time()
print(f"urllib read 15MB in {t1-t0:.2f}s ({(15/(t1-t0)):.2f} MB/s)")
